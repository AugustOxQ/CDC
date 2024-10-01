# #!/usr/bin/env python

# # In[ ]:


# import json
# import os
# from glob import glob

# from PIL import Image
# from tqdm import tqdm

# # In[ ]:


# def load_first_type(annotation_path, prefix):
#     # Load mscoco style annotation
#     with open(annotation_path) as f:
#         annotations = json.load(f)
#         for annotation in annotations:
#             annotation["image"] = os.path.join(prefix, annotation["image"])
#     return annotations


# # In[ ]:


# def load_second_type(folder_path, image_prefix):
#     # Load redcaps annotations
#     combined_annotations = []
#     # Load all json files in the folder
#     json_files = glob(os.path.join(folder_path, "*.json"))
#     for idx, json_file in enumerate(tqdm(json_files)):
#         with open(json_file) as f:
#             data = json.load(f)
#             # Two keys: "info" and "annotations", we only care about "annotations"
#             try:
#                 annotations = data["annotations"]
#             except:
#                 # Skip the file if it doesn't have the right keys
#                 print(f"Skipping file {json_file}")
#                 continue
#             # Get the subreddit name from the json file name
#             subreddit = os.path.basename(json_file).replace("_2020.json", "")
#             # Load the images from the subreddit folder
#             image_folder = os.path.join(image_prefix, subreddit)
#             for annotation in annotations:
#                 image_id = annotation["image_id"]
#                 image_path = os.path.join(image_folder, f"{image_id}.jpg")
#                 image_path_absolute = os.path.join("/data/PDD", image_path)
#                 if os.path.exists(image_path_absolute):
#                     try:
#                         # Try to open the image to ensure it's not corrupted
#                         Image.open(image_path_absolute)
#                         combined_annotations.append(
#                             {
#                                 "image": image_path,
#                                 "caption": annotation["caption"],
#                                 "image_id": annotation["image_id"],
#                             }
#                         )
#                     except (OSError, FileNotFoundError):
#                         print(f"Skipping corrupted or missing image: {image_path}")
#     return combined_annotations


# # In[ ]:


# def merge_annotations(first_annotations, second_annotations, output_path):
#     combined_annotations = first_annotations + second_annotations
#     with open(output_path, "w") as f:
#         json.dump(combined_annotations, f, indent=4)


# # In[ ]:


# def save_second_annotations_only(second_annotations, output_path):
#     with open(output_path, "w") as f:
#         json.dump(second_annotations, f, indent=4)


# # In[ ]:


# first_annotation_path = "/project/Deep-Clustering/data/flickr30k/train.json"
# first_prefix = "flickr30k/images"
# second_annotation_folder = "/data/PDD/redcaps/annotations2020"
# second_image_prefix = "redcaps/images2020"


# # In[ ]:


# # Save second_annotations to a file
# output_path = "/project/Deep-Clustering/data/redcaps_plus/redcaps.json"
# second_annotations = load_second_type(second_annotation_folder, second_image_prefix)
# print(f"Loaded {len(second_annotations)} annotations from {second_annotation_folder}")

# save_second_annotations_only(second_annotations, output_path)

# print(f"Second annotations saved to {output_path}")


# # In[ ]:


# output_path = "/project/Deep-Clustering/data/redcaps_plus/flickr30k_redcaps.json"
# first_annotations = load_first_type(first_annotation_path, first_prefix)
# print(f"Loaded {len(first_annotations)} annotations from {first_annotation_path}")
# merge_annotations(first_annotations, second_annotations, output_path)

# print(f"Combined annotations saved to {output_path}")


# # In[ ]:


# first_annotation_path = "/data/SSD/coco/annotations/coco_karpathy_train.json"
# first_prefix = "coco/images"
# output_path = "/project/Deep-Clustering/data/redcaps_plus/coco_redcaps.json"
# first_annotations = load_first_type(first_annotation_path, first_prefix)

# merge_annotations(first_annotations, second_annotations, output_path)

# print(f"Combined annotations saved to {output_path}")


# # In[ ]:


# # import json
# # import os
# # from PIL import Image
# # path = "/data/PDD/redcaps/annotations2020/abandoned_2020.json"


# # # In[ ]:


# # json_file = "abandoned_2020.json"
# # image_prefix = "redcaps/images2020"
# # with open(path) as f:
# #     data = json.load(f)
# # # Two keys: "info" and "annotations", we only care about "annotations"
# # annotations = data['annotations']
# # # Get the subreddit name from the json file name
# # subreddit = os.path.basename(json_file).replace('_2020.json', '')
# # # Load the images from the subreddit folder
# # image_folder = os.path.join(image_prefix, subreddit)
# # for annotation in annotations:
# #     image_id = annotation['image_id']
# #     image_path = os.path.join(image_folder, f"{image_id}.jpg")
# #     image_path_absolute = os.path.join("/data/PDD", image_path)
# #     print(image_path_absolute)
# #     if os.path.exists(image_path_absolute):
# #         try:
# #             # Try to open the image to ensure it's not corrupted
# #             Image.open(image_path_absolute)
# #         except (IOError, FileNotFoundError):
# #             print(f"Skipping corrupted or missing image: {image_path}")


# # In[ ]:
