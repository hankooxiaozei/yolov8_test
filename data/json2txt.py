import argparse
import json
import os

from tqdm import tqdm

keep_shape_labels = ["E"]


def convert_label_json(json_dir, save_dir, classes):
    classes = classes.split(",")
    # print("classes:" + classes)
    print("classes.num:" + str(len(classes)))
    # 获取指定目录下所有 .json 文件的列表
    json_paths = [f for f in os.listdir(json_dir) if f.lower().endswith(".json")]

    for json_path in tqdm(json_paths):
        # for json_path in json_paths:
        path = os.path.join(json_dir, json_path)
        # print(path)
        with open(path) as load_f:
            print(load_f)
            json_dict = json.load(
                load_f,
            )
        h, w = json_dict["imageHeight"], json_dict["imageWidth"]

        has_valid_shape = False
        for shape_dict in json_dict["shapes"]:
            label = shape_dict["label"]
            if label not in keep_shape_labels:
                continue
            shape_type = shape_dict["shape_type"]
            if shape_type != "polygon":
                continue
            has_valid_shape = True
            break

        if has_valid_shape:
            # save txt path
            txt_path = os.path.join(save_dir, json_path.replace("json", "txt"))
            txt_file = open(txt_path, "w")

            for shape_dict in json_dict["shapes"]:
                label = shape_dict["label"]
                if label not in keep_shape_labels:
                    continue
                shape_type = shape_dict["shape_type"]
                if shape_type != "polygon":
                    continue

                label_index = classes.index(label)
                points = shape_dict["points"]

                points_nor_list = []

                for point in points:
                    points_nor_list.append(point[0] / w)
                    points_nor_list.append(point[1] / h)

                points_nor_list = list(map(lambda x: str(x), points_nor_list))
                points_nor_str = " ".join(points_nor_list)

                label_str = str(label_index) + " " + points_nor_str + "\n"
                txt_file.writelines(label_str)
            txt_file.close()
        else:
            print(f"No valid shape found in {json_path}, skipping.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="json convert to txt params")
    parser.add_argument("--json-dir", type=str, default="dataset/json_labels", help="json path dir")
    parser.add_argument("--save-dir", type=str, default="dataset/labels", help="txt save dir")
    parser.add_argument("--classes", type=str, default="surface", help="classes")
    args = parser.parse_args()
    json_dir = args.json_dir
    save_dir = args.save_dir
    classes = args.classes

    # json_dir = "C:/Users/HL/Downloads/wendang_labels_1/images/"
    # # json_dir = "C:/Users/HL/Downloads/wendang_labels/images20250826/"
    # save_dir = "C:/Users/HL/Downloads/wendang_labels_1/label_test/"
    # classes = "E"
    convert_label_json(json_dir, save_dir, classes)
