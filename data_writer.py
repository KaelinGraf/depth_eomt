#replicator writer to save relevant scene information into a pre-determined dataset format
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from skimage.color import label2rgb
from omni.replicator.core import WriterRegistry
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer
from omni.replicator.core.scripts.functional import write_image, write_json
import cv2
import json
from pxr import UsdGeom, Gf, UsdPhysics,UsdShade,Sdf, Vt, Usd
from typing import Tuple
class ZividWriter(Writer):
    def __init__(self,output_dir,annotators):
        
       # annotators = ["CameraParams","occlusion","instance_segmentation",'bounding_box_3d']
        self.annotators = []
        self.data_structure = "renderProduct"
        self.backend = BackendDispatch(output_dir=output_dir)
        self.output_dir = output_dir
        self.frame_dir=None
        # #append required annotators
        for annotator in annotators:
            # if annotator == "instance_segmentation":
            #     self.annotators.append(AnnotatorRegistry.get_annotator(f"{annotator}",{"Colorise":True}))

            # else:
            self.annotators.append(AnnotatorRegistry.get_annotator(f"{annotator}"))
            
    def update_dir(self,frame_dir):
        self.frame_dir = frame_dir
    def get_dir(self):
        return self.frame_dir
    def write(self,data:dict):
        for rp_name, annotator_data in data["renderProducts"].items():
            print(f"{rp_name}:{annotator_data}")
            output = {}
            seg_data = annotator_data['instance_segmentation']
            segment_id_pairs = plot_replicator_instance_mask(seg_data,self.frame_dir,rp_name)
            cam_params = annotator_data['CameraParams']
            
            # Compute camera intrinsics
            w_res = cam_params["renderProductResolution"][0]
            h_res = cam_params["renderProductResolution"][1]
            pixel_size = cam_params["cameraAperture"][0] / w_res
            fx = float(cam_params["cameraFocalLength"] / pixel_size)
            fy = float(cam_params["cameraFocalLength"] / pixel_size)
            cx = float(w_res / 2.0 + cam_params["cameraApertureOffset"][0])
            cy = float(h_res / 2.0 + cam_params["cameraApertureOffset"][1])

            cam_K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            
            cam_K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            
            # --- Native World-to-Camera calculation bypassing Replicator entirely ---
            import omni.timeline
            from pxr import UsdGeom, Usd
            stage = omni.usd.get_context().get_stage()
            timeline = omni.timeline.get_timeline_interface()
            time_code = Usd.TimeCode(timeline.get_current_time() * stage.GetTimeCodesPerSecond())
            
            cam_prim_path = annotator_data.get('camera', '')
            if not cam_prim_path:
                print("Warning: 'camera' not found in annotator_data. Skipping.")
                continue
            cam_prim = stage.GetPrimAtPath(cam_prim_path)
            cam_l2w = np.array(UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(time_code))
            w2c_tf = np.linalg.inv(cam_l2w)
            
            # Transform USD Camera (-Z forward, +Y up) to OpenCV/BOP Camera (+Z forward, -Y down)
            T_usd_to_cv = np.array([
                [1.0,  0.0,  0.0, 0.0],
                [0.0, -1.0,  0.0, 0.0],
                [0.0,  0.0, -1.0, 0.0],
                [0.0,  0.0,  0.0, 1.0]
            ])
            w2c_tf_cv = np.matmul(w2c_tf, T_usd_to_cv)
            
            R_w2c_pure = w2c_tf_cv[:3, :3]
            t_w2c_pure = w2c_tf_cv[3, :3]
            cam_R_w2c = R_w2c_pure.T.flatten().tolist()
            cam_t_w2c = t_w2c_pure.tolist()
            # ------------------------------------------------------------------------

            objects_list = []
            output = {
                "camera": {
                    "cam_K": cam_K,
                    "resolution": [int(w_res), int(h_res)],
                    "cam_R_w2c": cam_R_w2c,
                    "cam_t_w2c": cam_t_w2c
                }
            }
            
            # Map object prim_paths to their bbox/pose/occlusion data
            path_to_bbox_data = {}
            if 'bounding_box_3d' in annotator_data:
                bbox_data = annotator_data['bounding_box_3d']
                bbox_array = bbox_data.get('data', [])
                prim_paths = bbox_data.get('info', {}).get('primPaths', [])
                if not prim_paths and 'primPaths' in bbox_data: # handle variations
                    prim_paths = bbox_data.get('primPaths', [])
                for b_idx, prim_path in enumerate(prim_paths):
                    if b_idx < len(bbox_array):
                        path_to_bbox_data[prim_path] = bbox_array[b_idx]

            for index, (str_id, semantic_label) in enumerate(seg_data['idToSemantics'].items()):
                id_int = int(str_id)
                object_dict = dict() # type: ignore
                prim_path = seg_data['idToLabels'].get(str_id, "")
                
                if "table_xform" in prim_path:
                    continue
                
                try:
                    seg_label = segment_id_pairs[id_int] #returns int 0-n of value in seg image
                except:
                    seg_label = int(-1)
                
                import re
                raw_class = str(semantic_label['class'])
                clean_class = re.sub(r'_instance_\d+$', '', raw_class)
                
                object_dict['class'] = clean_class  # type: ignore
                object_dict['prim_path'] = prim_path            # type: ignore
                object_dict['segmentation_id'] = int(seg_label) # type: ignore
                
                pose = None
                occlusion = -1.0
                
                import omni.usd
                from pxr import UsdGeom, Usd
                
                l2w_tf = None
                
                # Check bounding box data first seamlessly
                # Retrieve exact Local-to-World transform precisely at the active simulation tick
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
                    continue
                l2w_tf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time_code))
                
                # --------- BBOX COM ORIGIN SHIFT ---------
                # Mathematically shift the true local origin precisely to the Object-Aligned Bounding Box Center
                bbox_cache = UsdGeom.BBoxCache(time_code, ['default', 'render'])
                local_bound = bbox_cache.ComputeUntransformedBound(prim)
                
                if not local_bound.GetBox().IsEmpty():
                    centroid = local_bound.ComputeCentroid()
                    # A Local pure translation matrix 
                    centroid_tf = np.eye(4)
                    centroid_tf[3, :3] = [centroid[0], centroid[1], centroid[2]]
                    
                    # Apply local offset before applying the world transform
                    l2w_tf = np.matmul(centroid_tf, l2w_tf)
                # -----------------------------------------
                
                # Math derivation of Local to Camera transform (row-major: L2W @ W2C)
                loc2cam_tf_usd = np.matmul(l2w_tf, w2c_tf)
                
                # Transform to BOP/OpenCV coordinate system (+Z forward, -Y down)
                loc2cam_tf = np.matmul(loc2cam_tf_usd, T_usd_to_cv)
                
                # Derive precise exact fractional MLP occlusion via native array coverage
                px_count_vis = int(np.count_nonzero(seg_data['data'] == id_int))
                if px_count_vis <= 0:
                    continue
                    
                # -------------------------------------------------------------
                # Accurate Visibility via Triangle Rasterization
                # Rasterize each mesh face into a silhouette mask to get the
                # exact unoccluded pixel footprint (replaces convex hull approach
                # which inflates area for concave objects).
                # -------------------------------------------------------------
                fx, fy, cx, cy = cam_K[0], cam_K[4], cam_K[2], cam_K[5]
                silhouette = np.zeros((int(h_res), int(w_res)), dtype=np.uint8)
                
                for mesh_prim in Usd.PrimRange(prim):
                    if not mesh_prim.IsA(UsdGeom.Mesh):
                        continue
                    mesh = UsdGeom.Mesh(mesh_prim)
                    pts = mesh.GetPointsAttr().Get()
                    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
                    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
                    if not pts or not face_vertex_indices:
                        continue
                    
                    # Transform vertices to camera space (OpenCV convention)
                    mesh_l2w = np.array(UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(time_code))
                    mesh_l2c = np.matmul(np.matmul(mesh_l2w, w2c_tf), T_usd_to_cv)
                    
                    pts_np = np.array(pts)
                    hom_pts = np.hstack((pts_np, np.ones((pts_np.shape[0], 1))))
                    cam_pts = np.matmul(hom_pts, mesh_l2c)[:, :3]
                    
                    # Project all vertices to image plane
                    z = cam_pts[:, 2]
                    valid = z > 0.01  # In front of camera (OpenCV: +Z forward)
                    uv = np.zeros((len(pts_np), 2), dtype=np.float32)
                    uv[valid, 0] = (cam_pts[valid, 0] / z[valid]) * fx + cx
                    uv[valid, 1] = (cam_pts[valid, 1] / z[valid]) * fy + cy
                    
                    # Rasterize each face as a filled polygon into the silhouette
                    idx_offset = 0
                    for face_size in face_vertex_counts:
                        face_indices = list(
                            face_vertex_indices[idx_offset : idx_offset + int(face_size)]
                        )
                        idx_offset += int(face_size)
                        
                        # Skip faces with any vertex behind camera
                        if not all(valid[fi] for fi in face_indices):
                            continue
                        
                        face_uvs = uv[face_indices].astype(np.int32)
                        cv2.fillPoly(silhouette, [face_uvs], 255)
                
                full_area = int(np.count_nonzero(silhouette))
                visibility_ratio = 1.0
                if full_area > 0:
                    visibility_ratio = max(0.0, min(1.0, float(px_count_vis) / full_area))
                
                occlusion_ratio = 1.0 - visibility_ratio
                
                R_row = loc2cam_tf[:3, :3]
                t_cam = loc2cam_tf[3, :3]
                
                # Extract scale 
                scale_x = np.linalg.norm(R_row[0, :])
                scale_y = np.linalg.norm(R_row[1, :])
                scale_z = np.linalg.norm(R_row[2, :])
                
                # Prevent division by zero
                scales = np.array([scale_x, scale_y, scale_z])
                scales[scales == 0] = 1.0
                
                # Remove scale to get pure rotation
                R_pure_row = R_row / scales[:, np.newaxis]
                # BOP format uses column-major vectors (P_cam = R * P_loc + t), so transpose
                R_bop = R_pure_row.T
                
                pose = {
                    "cam_R_m2c": R_bop.flatten().tolist(),
                    "cam_t_m2c": t_cam.tolist(),
                    "scale_m2c": scales.tolist()
                }
                
                object_dict['pose'] = pose                                     # type: ignore
                object_dict['visibility_ratio'] = visibility_ratio             # type: ignore
                object_dict['occlusion_ratio'] = occlusion_ratio               # type: ignore
                objects_list.append(object_dict)
                
            output['objects'] = objects_list # type: ignore
                
            if self.frame_dir:
                json_path = f"{self.frame_dir}/{rp_name}_scene_info.json"
                with open(json_path, 'w') as f:
                    json.dump(output, f, indent=4)



def plot_replicator_instance_mask(replicator_data,output_dir,rp_name):
    mask_32 = np.asarray(replicator_data['data'],dtype=np.uint16) #dimensions h,w, single channel containing instance id
    diff_labels = np.unique(mask_32) #find allinduvidulabeprint(index,label in enumerate(diff_labels))    # Debug: show enumerated labels (avoid using `index`/`label` before they're defined)
    index_label_pairs = {}
    for index, label in enumerate(diff_labels): index_label_pairs[label]=index #store raw value LUT
    #id_to_label = replicator_data['idToLabels']
    raw_save_path = f"{output_dir}/{rp_name}_instance_raw.png"
    cv2.imwrite(raw_save_path, mask_32)
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0.0, 1.0, len(diff_labels)))
    h,w = mask_32.shape
    rgb_canvas = np.zeros((h,w,3))
    for index, label in enumerate(diff_labels):
        colour = colors[index][:3]
        rgb_canvas[mask_32==label] = colour
        
    plt.figure(figsize=(10, 6))
    plt.imshow(rgb_canvas)
    plt.axis('off') # Hide axis numbers
    
    # Save to disk instead of rendering to a screen
    plt.savefig(f"{output_dir}/{rp_name}_instance_output.png", bbox_inches='tight')
    plt.close()
    
    return index_label_pairs
        

def get_world_transform_xform(prim: Usd.Prim) -> Tuple[Gf.Vec3d, Gf.Rotation, Gf.Vec3d]:
    """
    Get the local transformation of a prim using Xformable.
    See https://openusd.org/release/api/class_usd_geom_xformable.html
    Args:
        prim: The prim to calculate the world transformation.
    Returns:
        A tuple of:
        - Translation vector.
        - Rotation quaternion, i.e. 3d vector plus angle.
        - Scale vector.
    """
    xform = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default() # The time at which we compute the bounding box
    world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
    translation: Gf.Vec3d = world_transform.ExtractTranslation()
    rotation: Gf.Rotation = world_transform.ExtractRotation()
    scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in world_transform.ExtractRotationMatrix()))
    
    #reconstruct as 4x4 matrix (without scale)
    matrx_no_scale = Gf.Matrix4d().SetTransform(rotation,translation)
    return translation, rotation, scale    

def get_6d_pose_cam_view(prim:Usd.Prim,t_cam:Gf.Matrix4d)-> Tuple[Gf.Vec3d,Gf.Rotation]:
    """
    Gets both the prim and camera view matrices, strips out the scale and returns a 4x4 pose matrix of Tobj->cam 

    Args:
        prim (Usd.Prim): Prim to find pose of (call stage.GetPrimAtPath(path) if given in path form)
        t_cam (Gf.Matrix4d): Tcam->world^-1 (i.e world->cam). typically retrieved from camera params annotator

    Returns:
        Gf.Matrix4d: resulting Tobj->cam
    """
    xform = UsdGeom.Xformable(prim)
    time = Usd.TimeCode.Default() # The time at which we compute the bounding box
    world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
    t_obj_cam : Gf.matrix4d = world_transform * t_cam
    translation: Gf.Vec3d = t_obj_cam.ExtractTranslation()
    rotation: Gf.Rotation = t_obj_cam.ExtractRotation()
    return (translation,rotation)
    
    
    
        
        
def register_writer():
    WriterRegistry.register(ZividWriter)
    (
        WriterRegistry._default_writers.append("ZividWriter")
        if "ZividWriter" not in WriterRegistry._default_writers
        else None
    )