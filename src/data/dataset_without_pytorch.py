"""Reference: https://github.com/MedMNIST/MedMNIST"""
from pathlib import Path
import random
import numpy as np
from PIL import Image
from medmnist.info import INFO, HOMEPAGE, DEFAULT_ROOT

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence


class MedMNIST(Sequence):

    flag = ...

    def __init__(self, split, transform=None, target_transform=None,
                 download=False, as_rgb=False, root=DEFAULT_ROOT):
        """
        dataset

        Parameters
        ----------
        split : str, possible are 'train', 'val' or 'test'
            Select a specific subset.
        transform : Callable, default = None
            Data transformation.
        target_transform : Callable, default = None
            Target transformation.
        download : bool, default = False
            Download the dataset.
        as_rgb : bool, default = False
            Return the dataset in RGB format.
        """

        self.info = INFO[self.flag]

        if root is not None and Path(root).exists():
            self.root = Path(root)
        else:
            raise RuntimeError(
                "Failed to setup the default `root` directory. "
                "Please specify and create the `root` directory manually.")

        if download:
            self.download()

        if not (self.root / f"{self.flag}.npz").exists():
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it')

        npz_file = np.load(str(self.root / f"{self.flag}.npz"))

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.as_rgb = as_rgb

        if self.split == 'train':
            self.imgs = npz_file['train_images']
            self.labels = npz_file['train_labels']
        elif self.split == 'val':
            self.imgs = npz_file['val_images']
            self.labels = npz_file['val_labels']
        elif self.split == 'test':
            self.imgs = npz_file['test_images']
            self.labels = npz_file['test_labels']
        else:
            raise ValueError

    def __len__(self):
        return self.imgs.shape[0]

    def __repr__(self):
        """Adapted from torchvision.ss"""
        _repr_indent = 4
        head = f"Dataset {self.__class__.__name__} ({self.flag})"
        body = [f"Number of datapoints: {self.__len__()}",
                f"Root location: {self.root}", f"Split: {self.split}",
                f"Task: {self.info['task']}",
                f"Number of channels: {self.info['n_channels']}",
                f"Meaning of labels: {self.info['label']}",
                f"Number of samples: {self.info['n_samples']}",
                f"Description: {self.info['description']}",
                f"License: {self.info['license']}"]

        lines = [head] + [" " * _repr_indent + line for line in body]
        return '\n'.join(lines)

    def download(self):
        """Download the dataset from the internet."""
        try:
            from torchvision.datasets.utils import download_url
            download_url(url=self.info["url"], root=str(self.root),
                         filename=f"{self.flag}.npz", md5=self.info["MD5"])
        except:
            raise RuntimeError(
                f"Something went wrong when downloading! Go to the homepage "
                f"to download manually. {HOMEPAGE}")

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        for x, y in data:
            xs.append(np.array(x))
            ys.append(y)
        return np.array(xs), np.array(ys)


class MedMNIST2D(MedMNIST):

    def __getitem__(self, index):
        """
        Typical getitem function.

        Returns
        -------
        (without transform/target_transform)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        """
        img, target = self.imgs[index], self.labels[index].astype(int)
        img = Image.fromarray(img)

        if self.as_rgb:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def save(self, folder, postfix="png", write_csv=True):
        """
        Save the data in a specific file format.

        Parameters
        ----------
        folder : str
            Destination to store the data at.
        postfix : str, default = "png"
            File suffix.
        write_csv : bool, default = True
            Save as CSV file.
        """
        from medmnist.utils import save2d
        save2d(imgs=self.imgs, labels=self.labels,
               img_folder=Path(folder) / self.flag, split=self.split,
               postfix=postfix, csv_path=(Path(folder) / f"{self.flag}.csv"
                                          if write_csv else None))

    def montage(self, length=20, replace=False, save_folder=None):
        """
        Visualize the data in multiple subplots as a montage.

        Parameters
        ----------
        length : int, default = 20
            Row and column length.
        replace : bool, default = False
            Select representative images without replacement.
        save_folder : str, default = None
            Location to save the collage at if not None.
        """
        from medmnist.utils import montage2d

        n_sel = length * length
        sel = np.random.choice(self.__len__(), size=n_sel, replace=replace)

        montage_img = montage2d(
            imgs=self.imgs, n_channels=self.info['n_channels'], sel=sel)

        if save_folder is not None:
            Path(save_folder).mkdir(parents=True, exist_ok=True)
            montage_img.save(Path(save_folder) /
                             f"{self.flag}_{self.split}_montage.jpg")
        return montage_img


class MedMNIST3D(MedMNIST):

    def __getitem__(self, index):
        """
        Typical getitem function.

        Returns
        -------
        (without transform/target_transform)
            img: array of 1x28x28x28 or 3x28x28x28 (if `as_RGB=True`), in [0,1]
            target: np.array of `L` (L=1 for single-label)
        """
        img, target = self.imgs[index], self.labels[index].astype(int)

        img = np.stack([img/255.]*(3 if self.as_rgb else 1), axis=0)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def save(self, folder, postfix="gif", write_csv=True):
        """
        Save the data in a specific file format.

        Parameters
        ----------
        folder : str
            Destination to store the data at.
        postfix : str, default = "gif"
            File suffix.
        write_csv : bool, default = True
            Save as CSV file.
        """
        from medmnist.utils import save3d

        assert postfix == "gif"

        save3d(imgs=self.imgs, labels=self.labels,
               img_folder=Path(folder) / self.flag, split=self.split,
               postfix=postfix, csv_path=(Path(folder) / f"{self.flag}.csv"
                                          if write_csv else None))

    def montage(self, length=20, replace=False, save_folder=None):
        """
        Visualize the data in multiple subplots as a montage.

        Parameters
        ----------
        length : int, default = 20
            Row and column length.
        replace : bool, default = False
            Select representative images without replacement.
        save_folder : str, default = None
            Location to save the collage at if not None.
        """
        assert self.info['n_channels'] == 1

        from medmnist.utils import montage3d, save_frames_as_gif
        n_sel = length * length
        sel = np.random.choice(self.__len__(), size=n_sel, replace=replace)

        montage_frames = montage3d(
            imgs=self.imgs, n_channels=self.info['n_channels'], sel=sel)

        if save_folder is not None:
            if not Path(save_folder).exists():
                Path(save_folder).mkdir(parents=True, exist_ok=True)

            save_frames_as_gif(
                montage_frames,
                Path(save_folder) / f"{self.flag}_{self.split}_montage.gif")

        return montage_frames


class PathMNIST(MedMNIST2D):
    flag = "pathmnist"


class OCTMNIST(MedMNIST2D):
    flag = "octmnist"


class PneumoniaMNIST(MedMNIST2D):
    flag = "pneumoniamnist"


class ChestMNIST(MedMNIST2D):
    flag = "chestmnist"


class DermaMNIST(MedMNIST2D):
    flag = "dermamnist"


class RetinaMNIST(MedMNIST2D):
    flag = "retinamnist"


class BreastMNIST(MedMNIST2D):
    flag = "breastmnist"


class BloodMNIST(MedMNIST2D):
    flag = "bloodmnist"


class TissueMNIST(MedMNIST2D):
    flag = "tissuemnist"


class OrganAMNIST(MedMNIST2D):
    flag = "organamnist"


class OrganCMNIST(MedMNIST2D):
    flag = "organcmnist"


class OrganSMNIST(MedMNIST2D):
    flag = "organsmnist"


class OrganMNIST3D(MedMNIST3D):
    flag = "organmnist3d"


class NoduleMNIST3D(MedMNIST3D):
    flag = "nodulemnist3d"


class AdrenalMNIST3D(MedMNIST3D):
    flag = "adrenalmnist3d"


class FractureMNIST3D(MedMNIST3D):
    flag = "fracturemnist3d"


class VesselMNIST3D(MedMNIST3D):
    flag = "vesselmnist3d"


class SynapseMNIST3D(MedMNIST3D):
    flag = "synapsemnist3d"


# backward-compatible
OrganMNISTAxial = OrganAMNIST
OrganMNISTCoronal = OrganCMNIST
OrganMNISTSagittal = OrganSMNIST


def get_loader(dataset, batch_size):
    """
    Ge a load function

    Parameters
    ----------
    dataset : MedMNIST
        The dataset to be loaded.
    batch_size : int
        The batch size to be used.
    """
    total_size = len(dataset)
    print('Size', total_size)
    index_generator = shuffle_iterator(range(total_size))
    while True:
        data = []
        for _ in range(batch_size):
            idx = next(index_generator)
            data.append(dataset[idx])
        yield dataset._collate_fn(data)


def shuffle_iterator(iterator):
    """
    Shuffle the dataset and then iterate.

    Parameters
    ----------
    iterator : Iterable
    """
    # iterator should have limited size
    index = list(iterator)
    total_size = len(index)
    i = 0
    random.shuffle(index)
    while True:
        yield index[i]
        i += 1
        if i >= total_size:
            i = 0
            random.shuffle(index)
