import os

""" 警告，该程序将删除掉data文件中所有的文件，如果需要，请备份其中的文件之后再保存 """

def delete_files_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f"已删除文件：{file_path}")
            except OSError as e:
                print(f"删除文件时出错：{file_path} - {e}")


if __name__ == "__main__":
    folder_path = r"data"
    delete_files_in_folder(folder_path)
    print("已清除指定文件夹中的所有文件。")