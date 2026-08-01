from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import random

class PromptDataSet(Dataset):
    def __init__(self, train_low_light_path_list, val_low_light_path_list, train_over_exposure_path_list, val_over_exposure_path_list,
                 train_ir_low_contrast_path_list, val_ir_low_contrast_path_list, train_ir_noise_path_list, val_ir_noise_path_list,
                 phase="train", transform=None, samples_per_epoch=800, max_val_samples=80):
        self.phase = phase
        if phase == "train":
            self.paths = {
                'low_light_A': train_low_light_path_list[0],
                'low_light_B': train_low_light_path_list[1],

                'over_exposure_A': train_over_exposure_path_list[0],
                'over_exposure_B': train_over_exposure_path_list[1],

                'ir_low_contrast_A': train_ir_low_contrast_path_list[0],
                'ir_low_contrast_B': train_ir_low_contrast_path_list[1],

                'ir_noise_A': train_ir_noise_path_list[0],
                'ir_noise_B': train_ir_noise_path_list[1],
            }
        else:
            self.paths = {
                'low_light_A': val_low_light_path_list[0],
                'low_light_B': val_low_light_path_list[1],

                'over_exposure_A': val_over_exposure_path_list[0],
                'over_exposure_B': val_over_exposure_path_list[1],

                'ir_low_contrast_A': val_ir_low_contrast_path_list[0],
                'ir_low_contrast_B': val_ir_low_contrast_path_list[1],

                'ir_noise_A': val_ir_noise_path_list[0],
                'ir_noise_B': val_ir_noise_path_list[1],
            }
        self.transform = transform

        # The original code counted both A and B lists, so one paired dataset
        # contributed 2*N items. Passing the same MSRS root to four task slots
        # then inflated an epoch to 8*N random samples (8664 for full MSRS).
        # Keep only genuinely different paired datasets and control the number
        # of optimizer updates independently from the dataset size.
        self.task_keys = []
        self.task_lengths = {}
        seen_datasets = set()
        for task_key in ('low_light', 'over_exposure', 'ir_low_contrast', 'ir_noise'):
            paths = (
                self.paths[task_key + '_A'],
                self.paths[task_key + '_B'],
            )
            lengths = [len(path_list) for path_list in paths]
            if len(set(lengths)) != 1:
                raise ValueError("Unpaired {} dataset lengths: {}".format(task_key, lengths))
            if lengths[0] == 0:
                continue

            signature = tuple(tuple(path_list) for path_list in paths)
            if signature in seen_datasets:
                continue
            seen_datasets.add(signature)
            self.task_keys.append(task_key)
            self.task_lengths[task_key] = lengths[0]

        if not self.task_keys:
            raise ValueError("No image pairs were found for phase '{}'".format(phase))

        self.num_unique_pairs = sum(self.task_lengths.values())
        self.samples_per_epoch = int(samples_per_epoch)
        all_val_samples = [
            (task_key, image_index)
            for task_key in self.task_keys
            for image_index in range(self.task_lengths[task_key])
        ]
        max_val_samples = int(max_val_samples)
        self.val_samples = (
            all_val_samples[:max_val_samples]
            if max_val_samples > 0
            else all_val_samples
        )

    def __len__(self):
        if self.phase == "train":
            if self.samples_per_epoch > 0:
                return self.samples_per_epoch
            return self.num_unique_pairs
        else:
            return len(self.val_samples)

    def __getitem__(self, item):
        if self.phase == "train":
            # Preserve the original random-with-replacement behavior while
            # keeping the number of updates equal for small and full MSRS.
            task_key = random.choice(self.task_keys)
            image_index = random.randrange(self.task_lengths[task_key])
        else:
            task_key, image_index = self.val_samples[item]

        # Load the A and B images based on the class and index
        image_A_path = self.paths[task_key + '_A'][image_index]
        image_B_path = self.paths[task_key + '_B'][image_index]

        image_A = Image.open(image_A_path).convert(mode='RGB')
        image_B = Image.open(image_B_path).convert(mode='RGB')
        image_A_gt = image_A.copy()
        image_B_gt = image_B.copy()
        image_full = image_A.copy()

        # Apply any specified transformations
        if self.transform is not None:
            image_A, image_B, image_A_gt, image_B_gt, image_full = self.transform(image_A, image_B, image_A_gt, image_B_gt, image_full)

        name = image_A_path.replace("\\", "/").split("/")[-1].split(".")[0]

        return image_A, image_B, image_A_gt, image_B_gt, image_full, task_key, name

    @staticmethod
    def collate_fn(batch):
        images_A, images_B, images_A_gt, images_B_gt, images_full, class_keys, name = zip(*batch)
        images_A = torch.stack(images_A, dim=0)
        images_B = torch.stack(images_B, dim=0)
        images_A_gt = torch.stack(images_A_gt, dim=0)
        images_B_gt = torch.stack(images_B_gt, dim=0)
        images_full = torch.stack(images_full, dim=0)
        return images_A, images_B, images_A_gt, images_B_gt, images_full, class_keys, name
