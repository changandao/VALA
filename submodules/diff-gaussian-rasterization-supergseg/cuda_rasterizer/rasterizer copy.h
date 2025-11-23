/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* superpoint_language_feature_precomp,
			const int* superpoint_id_precomp,
			const float* instance_feature_precomp,
			const float* hierarchical_feature_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* all_map,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* out_language_feature,
			float* out_instance_feature,
			float* out_hierarchical_feature,
			float* out_alpha,
			float* out_all_map,
			float* out_plane_depth,
			int* radii = nullptr,
			int* out_observe = nullptr,
			bool debug = false,
			bool include_language_feature = false,
			bool include_instance_feature = false,
			bool include_hierarchical_feature = false,
			const bool render_geo = false);


		static void visible_filter(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int M,
			const int width, int height,
			const float* means3D,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			int* radii,
			bool debug);
		
		
		
		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const float* all_map_pixels,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* superpoint_language_feature_precomp,
			const int* superpoint_id_precomp,
			const float* instance_feature_precomp,
			const float* hierarchical_feature_precomp,
			const float* all_maps,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			const float* dL_dpix_language_feature,
			const float* dL_dpix_instance_feature,
			const float* dL_dpix_hierarchical_feature,
			const float* dL_dout_all_map,
			const float* dL_dout_plane_depth,
			float* dL_dmean2D,
			float* dL_dmean2D_abs,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dsuperpoint_language_feature,
			float* dL_dinstance_feature,
			float* dL_dhierarchical_feature,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			float* dL_dall_map,
			bool debug,
			bool include_language_feature,
			bool include_instance_feature,
			bool include_hierarchical_feature,
			const bool render_geo);
	};
};

#endif