import os
import copy
import csv
import nltk
import numpy as np

import torch
import torch.utils.data as data


def get_dataset(data_path, data_name, data_split, vocab, return_id_caps=False):

    data_path = os.path.join(data_path, data_name)

    captions = []

    if data_name == "cc152k_precomp":
        img_ids = []
        with open(os.path.join(data_path, "%s_caps.tsv" % data_split)) as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for line in tsvreader:
                captions.append(line[1].strip())
                img_ids.append(line[0])

    elif data_name in ["coco_precomp", "f30k_precomp"]:
        with open(os.path.join(data_path, "%s_caps.txt" % data_split), "r") as f:
            for line in f:
                captions.append(line.strip())

    else:
        raise NotImplementedError("Unsupported dataset!")

    captions_token = []
    for index in range(len(captions)):
        caption = captions[index]
        tokens = nltk.tokenize.word_tokenize(caption.lower())

        caption = []
        caption.append(vocab("<start>"))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab("<end>"))

        captions_token.append(caption)

    images = np.load(os.path.join(data_path, "%s_ims.npy" % data_split))
    print(
        "from '{}' load {} data: {} images, {} captions".format(
            data_path, data_split, images.shape[0], len(captions)
        )
    )

    if return_id_caps:
        return captions_token, images, img_ids, captions
    else:
        return captions_token, images



class SamplesDataSet(data.Dataset):
    def __init__(
        self,
        captions,
        images,
        data_split,
        noise_ratio=0,
        noise_file="",
        mode="",
        mask=[],
        probability_A=[],
        probability_B=[],
    ):

        assert 0 <= noise_ratio <= 1

        self.captions = captions
        self.images = images
        self.data_split = data_split
        self.noise_ratio = noise_ratio

        self.mode = mode
        self.mask = mask
        self.probability_A=probability_A
        self.probability_B=probability_B

        self.length = len(self.captions)
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        if data_split == "dev":
            self.length = 1000 * self.im_div

        self.t2i_index = np.arange(0, self.length) // self.im_div

        if data_split == "train":
            self._t2i_index = copy.deepcopy(self.t2i_index)

            if noise_ratio:

                if os.path.exists(noise_file):
                    print("=> load noisy index from {}".format(noise_file))
                    self.t2i_index = np.load(noise_file)
                else:
                    idx = np.arange(0, self.length)
                    np.random.shuffle(idx)
                    noise_length = int(noise_ratio * self.length)


                    shuffle_index = self.t2i_index[idx[:noise_length]]
                    np.random.shuffle(shuffle_index)

                    self.t2i_index[idx[:noise_length]] = shuffle_index

                    np.save(noise_file, self.t2i_index)
                    print("=> save noisy index to {}".format(noise_file))


            self._labels = np.ones((self.length), dtype="int")
            self._labels[self._t2i_index != self.t2i_index] = 0

            split_idx = None
            if self.mode == "strictly_clean" or self.mode == "clean":

                split_idx = mask.nonzero()[0]
                self.probability_A = [probability_A[i] for i in split_idx]
                self.probability_B = [probability_B[i] for i in split_idx]

            elif self.mode == "noisy":
                split_idx = mask.nonzero()[0]

            if split_idx is not None:
                self.captions = [self.captions[i] for i in split_idx]
                self.t2i_index = [self.t2i_index[i] for i in split_idx]
                self._t2i_index = [self._t2i_index[i] for i in split_idx]
                self._labels = [self._labels[i] for i in split_idx]
                self.length = len(self.captions)

        print("{} {} data has a size of {}".format(data_split, self.mode, self.length))


    def __getitem__(self, index):

        image = torch.Tensor(self.images[self.t2i_index[index]])
        text = torch.Tensor(self.captions[index])

        if self.data_split == "train":
            if self.mode == "strictly_clean" or self.mode == "clean":
                return (
                    image,
                    text,
                    index,
                    torch.Tensor([1]),
                    torch.Tensor([self.probability_A[index]]),
                    torch.Tensor([self.probability_B[index]]),
                    self._labels[index],
                )
            elif self.mode == "noisy":
                return image, text, index, self._labels[index], 0

            else:
                return image, text, index, self.t2i_index[index]
        else:
            return image, text, index, self.t2i_index[index]

    def __len__(self):

        return self.length

def collate_fn(data):


    labels = None

    if len(data[0]) == 7:

        images, captions, ids, labels, prob_A, prob_B, _labels = zip(*data)
        # Merge
        labels = torch.stack(labels, 0).long()
        # Merge
        prob_A = torch.stack(prob_A, 0)
        prob_B = torch.stack(prob_B, 0)

    elif len(data[0]) == 5:
        images, captions, ids, _labels, labels = zip(*data)

    else:

        images, captions, ids, img_ids = zip(*data)

    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    text = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        text[i, :end] = cap[:end]


    if len(data[0]) == 7:

        return images, text, lengths, ids, labels, prob_A, prob_B, _labels
    elif len(data[0]) == 5:
        return images, text, lengths, ids, labels, _labels

    else:
        return images, text, lengths, ids

def get_loader(
    captions,
    images,
    data_split,
    batch_size,
    workers,
    noise_ratio=0,
    noise_file="",
    strictly_clean=[],
    clean=[],
    noisy=[],
    prob_A=[],
    prob_B=[],
):


    if data_split == "warmup":
        dset = SamplesDataSet(
            captions,
            images,
            "train",
            noise_ratio,
            noise_file,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset=dset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
            num_workers=workers,
        )
        return data_loader, dset.length, dset._labels

    # 搞清楚，为什么要往里面传进prob_A prob_B
    elif data_split == "train":
        clean_dataset = SamplesDataSet(
            captions,
            images,
            "train",
            noise_ratio,
            noise_file,
            mode="strictly_clean",
            mask=strictly_clean,
            probability_A=prob_A,
            probability_B=prob_B
        )
        clean_trainloader = torch.utils.data.DataLoader(
            dataset=clean_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
            num_workers=workers,
        )

        hard_dataset = SamplesDataSet(
            captions,
            images,
            "train",
            noise_ratio,
            noise_file,
            mode="clean",
            mask=clean,
            probability_A=prob_A,
            probability_B=prob_B
        )
        hard_trainloader = torch.utils.data.DataLoader(
            dataset=hard_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
            num_workers=workers,
        )

        noisy_dataset = SamplesDataSet(
            captions,
            images,
            "train",
            noise_ratio,
            noise_file,
            mode="noisy",
            mask=noisy,
            probability_A=prob_A,
            probability_B=prob_B
        )
        noisy_trainloader = torch.utils.data.DataLoader(
            dataset=noisy_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
            num_workers=workers,
        )

        return clean_trainloader, hard_trainloader, noisy_trainloader

    elif data_split == "dev":
        dset = SamplesDataSet(captions, images, data_split)
        data_loader = torch.utils.data.DataLoader(
            dataset=dset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            num_workers=workers,
        )
    elif data_split in ["test", "testall", "test5k"]:
        dset = SamplesDataSet(captions, images, data_split)
        data_loader = torch.utils.data.DataLoader(
            dataset=dset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            num_workers=workers,
        )
    else:
        raise NotImplementedError("Not support data split!")
    return data_loader