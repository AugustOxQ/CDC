# class CDC_train(Dataset):
#     def __init__(self, annotation_path, image_path, preprocess, ratio=0.1):
#         self.annotations = json.load(open(annotation_path))
#         self.image_path = image_path
#         self.vis_processors = preprocess

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, idx):
#         raw_image = Image.open(
#             os.path.join(self.image_path, self.annotations[idx]["image"])
#         ).convert("RGB")
#         image_input = self.vis_processors(raw_image, return_tensors="pt")
#         if "pixel_values" in image_input:
#             image_input["pixel_values"] = image_input["pixel_values"].squeeze()

#         raw_text = self.annotations[idx]["caption"]

#         return image_input, raw_text


# class CC_train(Dataset):
#     def __init__(self, annotation_path, image_path=None, preprocess=None, ratio=0.1):
#         """Custom dataset class for loading images from a Hugging Face dataset.

#         Args:
#             dataset (datasets.Dataset): The dataset split containing images.
#             transform (callable, optional): Optional transform to be applied on a sample.
#         """

#         self.dataset = load_dataset(
#             "pixparse/cc3m-wds",
#             cache_dir=custom_download_path,
#             # split="train",
#             # split="train[:10%]+train[-80%:]",
#             split=f"train[:{ratio*100}%]",
#         )
#         self.preprocess = preprocess

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         # For coco its image, for CC12M its jpg
#         image = self.preprocess(images=self.dataset[idx]["jpg"], return_tensors="pt")

#         if "pixel_values" in image:
#             image["pixel_values"] = image["pixel_values"].squeeze()
#         raw_text = self.dataset[idx]["txt"]

#         return image, raw_text
