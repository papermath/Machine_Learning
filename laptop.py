import pandas as pd
import numpy as np
import re

laptop = pd.read_csv("laptopData.csv")

laptop.rename({"Memory": "Storage", "Ram": "RAM", "Inches": "Screen_Size"}, inplace = True, axis = 1)
laptop.replace("?", np.nan, inplace = True)
laptop.dropna(inplace = True)

laptop.reset_index(drop = True, inplace = True)

laptop.Screen_Size = pd.to_numeric(laptop.Screen_Size)
memory_split = laptop.Storage.str.split("+")

laptop["HDD"] = 0
laptop["SSD"] = 0
laptop["Flash_Storage"] = 0
laptop["Resolution"] = np.nan
laptop["Touchscreen"] = np.nan
laptop["IPS"] = np.nan
laptop["CPU"] = np.nan
laptop["Clock_Speed"] = 0.0
laptop["GPU"] = np.nan

laptop.Resolution = laptop.Resolution.astype("string")
laptop.Touchscreen = laptop.Touchscreen.astype("string")
laptop.IPS = laptop.IPS.astype("string")
laptop.CPU = laptop.CPU.astype("string")
laptop.GPU = laptop.GPU.astype("string")

split_list = []
for memory_list in memory_split:
    current_list = []
    for item in memory_list:
        current_list.append(re.split(r" ", item))
    split_list.append(current_list)

for i, double_item_list in enumerate(split_list):
    for item_list in double_item_list:
        if "HDD" in item_list:
            HDD_list = [re.findall(r"\d+", item) for item in item_list if re.search(r"\d+", item)]
            laptop.loc[i, "HDD"] += int(HDD_list[0][0])
            if [item for item in item_list if re.search(r"TB", item)]:
                laptop.loc[i,"HDD"] *= 1000
        elif "SSD" in item_list:
            SSD_list = [re.findall(r"\d+", item) for item in item_list if re.search(r"\d+", item)]
            laptop.loc[i, "SSD"] += int(SSD_list[0][0])
            if [item for item in item_list if re.search(r"TB", item)]:
                laptop.loc[i,"SSD"] *= 1000
        elif "Flash" in item_list:
            Flash_list = [re.findall(r"\d+",item) for item in item_list if re.search(r"\d+", item)]
            laptop.loc[i, "Flash_Storage"] += int(Flash_list[0][0])
            if [item for item in item_list if re.search(r"TB", item)]:
                laptop.loc[i, "Flash_Storage"] *= 1000
        else:
            print("item not found")

for i,resolution in enumerate(laptop.ScreenResolution):
    laptop.loc[i, "Resolution"] = re.search(r"\d+x\d+", resolution).group(0)
    if re.search(r"Touchscreen", resolution):
        laptop.loc[i, "Touchscreen"] = "Yes"
    else:
        laptop.loc[i, "Touchscreen"] = "No"
    if re.search(r"IPS", resolution):
        laptop.loc[i, "IPS"] = "Yes"
    else:
        laptop.loc[i, "IPS"] = "No"

for i, cpu in enumerate(laptop.Cpu):
    if re.search(r"Intel", cpu):
        if re.search(r"Intel Core i\d", cpu):
            laptop.loc[i, "CPU"] = re.search(r"Intel Core i\d", cpu).group(0).strip()
        else:
            core = re.sub(r"(\w?\d?-?\w*\d+\w* ?\w?\d?)? \d+.?\d*GHz","", cpu)
            laptop.loc[i, "CPU"] = core.strip()
    elif re.search(r"AMD", cpu):
        core = re.sub(r"(\w?\d?\d?-?\d*\w*)? \d.?\d?GHz", "", cpu)
        laptop.loc[i, "CPU"] = core.strip()
    laptop.loc[i, "Clock_Speed"] = float(re.search(r"[0-9.]*",re.search(r"[0-9.]*GHz", cpu).group(0)).group(0))

for i, ram, in enumerate(laptop.RAM):
    laptop.loc[i, "RAM"] = re.search(r"\d*", ram).group(0)

for i, gpu in enumerate(laptop.Gpu):
    if re.search(r"AMD", gpu):
        if re.search(r"R\d", gpu):
            laptop.loc[i, "GPU"] = re.search(r"(AMD Radeon R\d)|(AMD R\d)", gpu).group(0)
        else:
            laptop.loc[i, "GPU"] = re.sub(r"(\w?\d+\w?)", "", gpu)
    elif re.search(r"Intel", gpu):
        new_gpu = re.search(r"Intel[a-zA-Z ]*", gpu).group(0)
        laptop.loc[i, "GPU"] = new_gpu.strip()
    elif re.search(r"Nvidia", gpu):
        if re.search(r"Quadro", gpu):
            new_gpu = re.sub(r"M?\d+M?","", gpu)
            laptop.loc[i, "GPU"] = new_gpu.strip()
        else:
            new_gpu = re.search(r"Nvidia (GeForce )?(GTX)?", gpu).group(0)
            laptop.loc[i, "GPU"] = new_gpu.strip()

laptop.OpSys = laptop.OpSys.replace({"macOS": "MacOS", "Mac OS X" : "MacOS", "Windows 10": "Windows", "Windows 7":"Windows", "Windows 10 S": "Windows", "Chrome OS": "ChromeOS"})

for i, weight in enumerate(laptop.Weight):
    laptop.loc[i, "Weight"] = weight[:-2]

laptop.Weight = laptop.Weight.astype("float64")

laptop.drop(columns = ["Storage", "ScreenResolution", "Cpu", "Gpu", "Unnamed: 0"], inplace = True)

laptop.Price = np.round(laptop.Price * 0.012 * 1.05,2)

laptop = laptop[["Company","TypeName","Screen_Size","Resolution","Touchscreen","IPS","CPU","Clock_Speed","RAM","GPU","HDD","SSD","Flash_Storage","OpSys","Weight","Price"]]
laptop.to_csv("laptop.csv", index = False)


