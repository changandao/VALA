#!/usr/bin/env python
import torch
import torchvision
import open_clip


class OpenCLIPNetwork:
    def __init__(self, device):
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.clip_model_type = "ViT-B-16"
        self.clip_model_pretrained = 'laion2b_s34b_b88k'
        self.clip_n_dims = 512
        model, _, _ = open_clip.create_model_and_transforms(
            self.clip_model_type,
            pretrained=self.clip_model_pretrained,
            precision="fp16",
        )
        model.eval()
        
        self.tokenizer = open_clip.get_tokenizer(self.clip_model_type)
        self.model = model.to(device)

        self.negatives = ("object", "things", "stuff", "texture")
        self.positives = (" ",)
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to(device)
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to(device)
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        # embed: torch.Size([721240, 512]) -> (n_pixels, embed_dim), positive_id -> phrase id
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        # print(f'phrases_embeds: {phrases_embeds.shape}') # torch.Size([11, 512])
        # print(f'pos_embeds: {self.pos_embeds.shape}') # torch.Size([7, 512])
        # print(f'neg_embeds: {self.neg_embeds.shape}') # torch.Size([4, 512])
        p = phrases_embeds.to(embed.dtype)
        output = torch.mm(embed, p.T) # 721240x512 * 512x11 -> 721240x11
        positive_vals = output[..., positive_id : positive_id + 1] # (721240, 1) similarities between the rendered embeddings and the positive query phrase
        negative_vals = output[..., len(self.positives) :] # (721240, 4) similarities between the rendered embeddings and the negative query phrases [object, things,...]
        repeated_pos = positive_vals.repeat(1, len(self.negatives)) # (721240, 1) -> (721240, 4)

        sims = torch.stack((repeated_pos, negative_vals), dim=-1) # torch.Size([721240, 4, 2])
        softmax = torch.softmax(10 * sims, dim=-1) # torch.Size([721240, 4, 2])
        best_id = softmax[..., 0].argmin(dim=1) # torch.Size([721240])

        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
            :, 0, :
        ]

    def encode_image(self, input, mask=None):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input, mask=mask)

    def encode_text(self, text_list, device):
        text = self.tokenizer(text_list).to(device)
        return self.model.encode_text(text)
    
    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.positives]
                ).to(self.neg_embeds.device)
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
    
    def set_semantics(self, text_list):
        self.semantic_labels = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.semantic_labels]).to("cuda")
            self.semantic_embeds = self.model.encode_text(tok_phrases)
        self.semantic_embeds /= self.semantic_embeds.norm(dim=-1, keepdim=True)
    
    def get_semantic_map(self, sem_map: torch.Tensor) -> torch.Tensor:
        # embed: 3xhxwx512
        n_levels, h, w, c = sem_map.shape
        pos_num = self.semantic_embeds.shape[0]
        phrases_embeds = torch.cat([self.semantic_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(sem_map.dtype)
        sem_pred = torch.zeros(n_levels, h, w)
        for i in range(n_levels):
            output = torch.mm(sem_map[i].view(-1, c), p.T)
            softmax = torch.softmax(10 * output, dim=-1)
            sem_pred[i] = torch.argmax(softmax, dim=-1).view(h, w)
            sem_pred[i][sem_pred[i] >= pos_num] = -1
        return sem_pred.long()

    def get_max_across(self, sem_map): # sem_map: torch.Size([3, 731, 988, 512]) -> (granuity, h, w, embed_dim)
        '''
        processes a semantic map and returns a relevance map, 
        highlighting the regions of the input image that are most relevant to specific phrases.
        '''
        n_phrases = len(self.positives)
        n_phrases_sims = [None for _ in range(n_phrases)]
        
        n_levels, h, w, _ = sem_map.shape
        clip_output = sem_map.permute(1, 2, 0, 3).flatten(0, 1) # 3x731x988x512 -> 731x988x3x512 -> 721240x3x512

        n_levels_sims = [None for _ in range(n_levels)]
        for i in range(n_levels):
            for j in range(n_phrases):
                probs = self.get_relevancy(clip_output[..., i, :], j) # clip_output[..., i, :] -> 721240, j -> phrase id
                pos_prob = probs[..., 0:1] # pos_prob -> torch.Size([721240, 1])
                n_phrases_sims[j] = pos_prob # phrase's level relevance score
            n_levels_sims[i] = torch.stack(n_phrases_sims) # each granularity level's relevance score for all phrases
        
        relev_map = torch.stack(n_levels_sims).view(n_levels, n_phrases, h, w)
        return relev_map
    
    def get_max_across_3d(self, clip_output):
        n_phrases = len(self.positives)
        n_phrases_sims = [None for _ in range(n_phrases)]
        for j in range(n_phrases):
            probs = self.get_relevancy(clip_output, j)
            pos_prob = probs[..., 0:1]
            n_phrases_sims[j] = pos_prob
        relev_map = torch.stack(n_phrases_sims).squeeze()
        return relev_map