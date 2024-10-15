import random
import yaml
import pandas as pd
import cv2
from PIL import Image, ImageEnhance
from astropy.visualization import make_lupton_rgb
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from sklearn.preprocessing import MinMaxScaler
from utlis.dataset_split import *
from scipy.ndimage import rotate
from skimage.util import random_noise

class MyDataset(Dataset):
    def __init__(self, files, combination,hyp_dict=None,data_dict=None):
        self.hyp_dict=hyp_dict
        self.w, self.h = self.hyp_dict["size"][0], self.hyp_dict["size"][1]
        self.images_gri = []
        self.images_urz = []
        self.pm = []
        self.px = []
        self.labels = []
        self.radius = []
        self.ag = []
        self.hyp_dict = hyp_dict
        self.type=[]
        self.radius.extend(files['radius_flame'])
        self.labels.extend(files['subclass'])
        self.pm.extend(files['pm'])
        self.px.extend(files['parallax'])
        self.pm = self.normalize_labels_pm(np.array(self.pm))
        self.px = self.normalize_labels_px(np.array(self.px))
        self.type.extend(files['type'])
        n = np.array(files['n'])
        ##取图片的操作
        for i in range(len(n)):
            path1 = data_dict['img'] + str(n[i]) + '.npy'
            npy = np.load(path1)

            # gri = make_lupton_rgb(npy[:, :, 1], npy[:, :, 2], npy[:, :, 3], Q=8, stretch=0.5)  # irg
            # urz = make_lupton_rgb(npy[:, :, 0], npy[:, :, 2], npy[:, :, 4], Q=8, stretch=0.5)

            u, g, r, i, z = npy[:, :, 0], npy[:, :, 1], npy[:, :, 2], npy[:, :, 3], npy[:, :, 4] #uirgz

            if combination == 'gri+uz':
                gri = make_lupton_rgb(g, r, i, Q=8, stretch=0.5)
                urz = make_lupton_rgb(u, z, u / 2 + z / 2, Q=8, stretch=0.5)
            elif combination == 'gru+iz':
                gri = make_lupton_rgb(g, r, u, Q=8, stretch=0.5)
                urz = make_lupton_rgb(i, z, i / 2 + z / 2, Q=8, stretch=0.5)
            elif combination == 'grz+iu':
                gri = make_lupton_rgb(g, r, z, Q=8, stretch=0.5)
                urz = make_lupton_rgb(i, u, i / 2 + u / 2, Q=8, stretch=0.5)
            elif combination == 'giu+rz':
                gri = make_lupton_rgb(g, i, u, Q=8, stretch=0.5)
                urz = make_lupton_rgb(r, z, r / 2 + z / 2, Q=8, stretch=0.5)
            elif combination == 'giz+ru':
                gri = make_lupton_rgb(g, i, z, Q=8, stretch=0.5)
                urz = make_lupton_rgb(r, u, r / 2 + u / 2, Q=8, stretch=0.5)
            elif combination == 'guz+ri':
                gri = make_lupton_rgb(g, u, z, Q=8, stretch=0.5)
                urz = make_lupton_rgb(r, i, r / 2 + i / 2, Q=8, stretch=0.5)
            elif combination == 'riu+gz':
                gri = make_lupton_rgb(r, i, u, Q=8, stretch=0.5)
                urz = make_lupton_rgb(g, z, g / 2 + z / 2, Q=8, stretch=0.5)
            elif combination == 'riz+gu':
                gri = make_lupton_rgb(r, i, z, Q=8, stretch=0.5)
                urz = make_lupton_rgb(g, u, g / 2 + u / 2, Q=8, stretch=0.5)
            elif combination == 'ruz+gi':
                gri = make_lupton_rgb(r, u, z, Q=8, stretch=0.5)
                urz = make_lupton_rgb(g, i, g / 2 + i / 2, Q=8, stretch=0.5)
            elif combination == 'iuz+gr':
                gri = make_lupton_rgb(i, u, z, Q=8, stretch=0.5)
                urz = make_lupton_rgb(g, r, g / 2 + r / 2, Q=8, stretch=0.5)
            elif combination == 'gri+ruz':
                gri = make_lupton_rgb(g, r, i, Q=8, stretch=0.5)
                urz = make_lupton_rgb(r, u, z, Q=8, stretch=0.5)
            elif combination == 'gri+iuz':
                gri = make_lupton_rgb(g, r, i, Q=8, stretch=0.5)
                urz = make_lupton_rgb(i, u, z, Q=8, stretch=0.5)
            elif combination == 'gri+guz':
                gri = make_lupton_rgb(g, r, i, Q=8, stretch=0.5)
                urz = make_lupton_rgb(g, u, z, Q=8, stretch=0.5)
            elif combination == 'gru+giz':
                gri = make_lupton_rgb(g, r, u, Q=8, stretch=0.5)
                urz = make_lupton_rgb(g, i, z, Q=8, stretch=0.5)
            elif combination == 'gru+riz':
                gri = make_lupton_rgb(g, r, u, Q=8, stretch=0.5)
                urz = make_lupton_rgb(r, i, z, Q=8, stretch=0.5)
            elif combination == 'gru+iuz':
                gri = make_lupton_rgb(g, r, u, Q=8, stretch=0.5)
                urz = make_lupton_rgb(i, u, z, Q=8, stretch=0.5)
            elif combination == 'giu+grz':
                gri = make_lupton_rgb(g, i, u, Q=8, stretch=0.5)
                urz = make_lupton_rgb(g, r, z, Q=8, stretch=0.5)
            elif combination == 'giu+riz':
                gri = make_lupton_rgb(g, i, u, Q=8, stretch=0.5)
                urz = make_lupton_rgb(r, i, z, Q=8, stretch=0.5)
            elif combination == 'giu+ruz':
                gri = make_lupton_rgb(g, i, u, Q=8, stretch=0.5)
                urz = make_lupton_rgb(r, u, z, Q=8, stretch=0.5)
            elif combination == 'giz+riu':
                gri = make_lupton_rgb(g, i, z, Q=8, stretch=0.5)
                urz = make_lupton_rgb(r, i, u, Q=8, stretch=0.5)
            elif combination == 'giz+ruz':
                gri = make_lupton_rgb(g, i, z, Q=8, stretch=0.5)
                urz = make_lupton_rgb(r, u, z, Q=8, stretch=0.5)
            elif combination == 'guz+riu':
                gri = make_lupton_rgb(g, u, z, Q=8, stretch=0.5)
                urz = make_lupton_rgb(r, i, u, Q=8, stretch=0.5)
            elif combination == 'guz+riz':
                gri = make_lupton_rgb(g, u, z, Q=8, stretch=0.5)
                urz = make_lupton_rgb(r, i, z, Q=8, stretch=0.5)
            elif combination == 'riu+grz':
                gri = make_lupton_rgb(r, i, u, Q=8, stretch=0.5)
                urz = make_lupton_rgb(g, r, z, Q=8, stretch=0.5)
            elif combination == 'iuz+grz':
                gri = make_lupton_rgb(i, u, z, Q=8, stretch=0.5)
                urz = make_lupton_rgb(g, r, z, Q=8, stretch=0.5)
            else:
                gri = make_lupton_rgb(g, r, i, Q=8, stretch=0.5)
                #gri = make_lupton_rgb(r, r, r, Q=8, stretch=0.5)
                urz = make_lupton_rgb(u, r, z, Q=8, stretch=0.5)
            


            self.images_gri.append(gri)
            self.images_urz.append(urz)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        gri = self.images_gri[index]
        urz = self.images_urz[index]
        pm = self.pm[index]
        px = self.px[index]
        label = self.labels[index]
        radius = self.radius[index]
        if (self.type[index]=='True') | (self.type[index]==True):
            gri,urz=self.dataAugment(gri,urz)

        if self.hyp_dict["augment"]:
            if random.random() < self.hyp_dict["flipud"]:
                gri = np.flipud(gri)
                urz = np.flipud(urz)

            # flip left-right
            if random.random() < self.hyp_dict["fliplr"]:
                gri = np.fliplr(gri)
                urz = np.fliplr(urz)

            if random.random() < self.hyp_dict["rot90"]:
                for _ in range(0, np.random.randint(1, 3)):
                    gri = np.rot90(gri, 1, (0, 1))
                    urz = np.rot90(urz, 1, (0, 1))

            if random.random() <self.hyp_dict['cutout']:
                gri,urz=self.cutout_pair(gri,urz)

            if random.random() <self.hyp_dict['erase']:
                gri,urz=self.random_erasing_pair(gri,urz)

            # if random.random() < self.hyp_dict["brightness"]:
            #     if gri.shape[1] <= 20:
            #         brightness = 1 + 0.2 * np.random.random()
            #     else:
            #         brightness = 0.8 + 0.4 * np.random.random()
            #     gri = self.brightnessEnhancement(gri, brightness)
            #     urz = self.brightnessEnhancement(urz, brightness)

        gri = cv2.resize(gri, (self.w, self.h))
        urz = cv2.resize(urz, (self.w, self.h))

        gri = np.transpose(gri / 255, (2, 0, 1)).astype(np.float32)
        urz = np.transpose(urz / 255, (2, 0, 1)).astype(np.float32)

        return gri, urz, pm, px, label , radius



    def brightnessEnhancement(self, image, brightness):  # 亮度增强
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        enh_bri = ImageEnhance.Brightness(image)
        image_brightened = enh_bri.enhance(brightness)
        image_brightened = cv2.cvtColor(np.array(image_brightened), cv2.COLOR_RGB2BGR)
        return image_brightened

    def addNoise(self, gri,urz):
        import time
        seed=int(time.time())
        return random_noise(gri, mode='gaussian', seed=seed, clip=True),random_noise(urz, mode='gaussian', seed=seed, clip=True)

    def normalize_labels_pm(self, labels):
        # if scaler is None:
        #     scaler = MinMaxScaler()
        #     scaler.fit(labels.reshape(-1, 1))
        import pickle
        with open(".\scale\scaler_pm.pkl", 'rb') as file:
            scaler = pickle.load(file)
        normalized_labels = scaler.transform(labels.reshape(-1, 1))
        return normalized_labels

    def normalize_labels_px(self, labels):
        # if scaler is None:
        #     scaler = MinMaxScaler()
        #     scaler.fit(labels.reshape(-1, 1))
        import pickle
        with open(".\scale\scaler_px.pkl", 'rb') as file:
            scaler = pickle.load(file)
        normalized_labels = scaler.transform(labels.reshape(-1, 1))
        return normalized_labels

    def rotate_image(self, image, angle):
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated_image

    def cutout_pair(self, image1, image2, num_patches=1, patch_size=4):
        height, width = image1.shape[:2]

        for _ in range(num_patches):
            y = np.random.randint(height)
            x = np.random.randint(width)

            y1 = np.clip(y - patch_size // 2, 0, height)
            y2 = np.clip(y + patch_size // 2, 0, height)
            x1 = np.clip(x - patch_size // 2, 0, width)
            x2 = np.clip(x + patch_size // 2, 0, width)

            image1[y1:y2, x1:x2, :] = 0
            image2[y1:y2, x1:x2, :] = 0

        return image1, image2

    def random_erasing_pair(self, image1, image2, area_ratio_range=(0.02, 0.4), aspect_ratio_range=(0.3, 3)):
        """
        Applies the same random erasing augmentation to a pair of images.

        Parameters:
            image1, image2 (numpy.ndarray): Input images (should be of the same size).
            area_ratio_range (tuple): Min and max ratio of the erased area to the total area of the image.
            aspect_ratio_range (tuple): Min and max aspect ratio of the erased area.

        Returns:
            tuple: Tuple of augmented images.
        """
        height, width, _ = image1.shape
        area = height * width

        for _ in range(100):  # Try 100 times
            erase_area = random.uniform(area_ratio_range[0], area_ratio_range[1]) * area
            aspect_ratio = random.uniform(aspect_ratio_range[0], aspect_ratio_range[1])

            h = int(round(np.sqrt(erase_area * aspect_ratio)))
            w = int(round(np.sqrt(erase_area / aspect_ratio)))

            if w < width and h < height:
                x = random.randint(0, width - w)
                y = random.randint(0, height - h)
                image1[y:y + h, x:x + w, :] = np.random.randint(0, 255, (h, w, image1.shape[2]))
                image2[y:y + h, x:x + w, :] = np.random.randint(0, 255, (h, w, image2.shape[2]))
                return image1, image2

        return image1, image2

    def dataAugment(self, gri,urz):
        change_num = 0  # 改变的次数

        while change_num < 1:  # 默认至少有一种数据增强生效
            if random.random() < self.hyp_dict["rot90"]:  # 旋转
                change_num += 1
                random_integer = random.randint(0, 180)
                gri=self.rotate_image(gri,random_integer)
                urz=self.rotate_image(urz,random_integer)
            if random.random() < self.hyp_dict['noise']:
                if random.random() < self.hyp_dict['noise']:  # 加噪声
                    change_num += 1
                    gri,urz = self.addNoise(gri,urz)
            if random.random() < self.hyp_dict["flipud"]:
                change_num+=1
                gri = np.flipud(gri)
                urz = np.flipud(urz)
            if random.random() < self.hyp_dict["fliplr"]:
                change_num+=1
                gri = np.fliplr(gri)
                urz = np.fliplr(urz)
            if random.random() <self.hyp_dict['cutout']:
                change_num+=1
                gri,urz=self.cutout_pair(gri,urz)
            if random.random() <self.hyp_dict['erase']:
                change_num+=1
                gri,urz=self.random_erasing_pair(gri,urz)

        return gri,urz

def create_dataset(files, hyp_dict,com,data_dict):
    dataset = MyDataset(files,com,  hyp_dict,data_dict)
    return dataset


def create_dataloader(dataset, hyp_dict):
    batch_size = min(hyp_dict['batch_size'], len(dataset))

    dataloader = DataLoader(dataset=dataset,
                            num_workers=hyp_dict['workers'],
                            shuffle=True,
                            batch_size=batch_size,
                            pin_memory=True)
    return dataloader


def get_train_dataloader(data_dict, hyp_dict,com):
    train_path = data_dict['train']
    val_path = data_dict['val']

    train_files=pd.read_csv(train_path)
    val_files=pd.read_csv(val_path)

    label_map = {'A': 0, 'F': 1, 'G': 2, 'K': 3, 'M': 4}
    label_map_reverse = {v: k for k, v in label_map.items()}
    train_files['subclass'] = train_files['subclass'].map(label_map)
    val_files['subclass'] = val_files['subclass'].map(label_map)

    train_dataset = create_dataset(train_files,  hyp_dict,com,data_dict)
    val_dataset = create_dataset(val_files,  hyp_dict,com,data_dict)

    # if val_path is None:
    #     # 划分数据集
    #     train_size = int(0.8 * len(train_dataset))
    #     val_size = len(train_dataset) - train_size
    #     train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    # else:
    #     if os.path.isfile(train_path):
    #         val_files = open(train_path).readlines()  # class file_path
    #     else:
    #         val_files = get_file_paths(train_path)
    #     val_dataset = create_dataset((val_files, dataset_save_method), names, hyp_dict)

    val_dataset.augment = False
    train_loader = create_dataloader(train_dataset, hyp_dict)
    val_loader = create_dataloader(val_dataset, hyp_dict)

    return train_loader, val_loader



