import argparse
import json
from pathlib import Path


class MedicalSolver(object):
    CLSNAMES = ['brain', 'liver', 'retinal']
    PHASES = ['train', 'test']
    GOOD_SPECIES = {'good'}
    IMAGE_SUFFIXES = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'}

    def __init__(self, root='./data/medical'):
        self.root = Path(root)
        self.meta_path = self.root / 'meta.json'
      
    def _relative_path(self, path):
        return path.relative_to(self.root).as_posix()

    def _list_images(self, directory):
        if not directory.is_dir():
            return []
        return sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES
        )


    def run(self):
        info = dict(train={}, test={})
        anomaly_samples = 0
        normal_samples = 0

        for cls_name in self.CLSNAMES:
            cls_dir = self.root / cls_name
            for phase in self.PHASES:
                phase_dir = cls_dir / phase
                cls_info = []
                species = sorted(
                    path.name for path in phase_dir.iterdir()
                    if path.is_dir()
                )

                for specie in species:
                    is_abnormal = specie not in self.GOOD_SPECIES
                    img_paths = self._list_images(phase_dir / specie)
                    mask_paths = self._list_images(cls_dir / 'ground_truth' / specie) if is_abnormal else []

                    if is_abnormal and len(img_paths) != len(mask_paths):
                        raise ValueError(
                            f'Image/mask count mismatch for {cls_name}/{phase}/{specie}: '
                            f'{len(img_paths)} images vs {len(mask_paths)} masks'
                        )

                    for idx, img_path in enumerate(img_paths):
                        info_img = dict(
                            img_path=self._relative_path(img_path),
                            mask_path=self._relative_path(mask_paths[idx]) if is_abnormal else '',
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)

                        if phase == 'test':
                            if is_abnormal:
                                anomaly_samples += 1
                            else:
                                normal_samples += 1

                info[phase][cls_name] = cls_info

        with open(self.meta_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(info, indent=4) + '\n')

        print('meta_path', self.meta_path)
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)


if __name__ == '__main__':
    runner = MedicalSolver(root='./data/medical')
    runner.run()
