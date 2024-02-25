import pandas as pd
from datetime import datetime


def file_load(file_path):
    dataset = {
        "ip": pd.read_csv(file_path + "ip.csv"),
        "access": pd.read_csv(file_path + "access.csv"),
        "flint": pd.read_csv(file_path + "flint.csv"),
        "fqdn": pd.read_csv(file_path + "fqdn.csv"),
        "ipv6": pd.read_csv(file_path + "ipv6.csv"),
        "label": pd.read_csv(file_path + "label.csv"),
        "label": pd.read_csv(file_path + "label.csv"),
        "whois": pd.read_json(file_path + "whois.json"),
        "all": pd.DataFrame(),
    }

    return dataset


file_path = "./dataset/dns1_demo/"
dataset = file_load(file_path)

# access merge ip
# acc_with_ip = pd.merge(dataset["ip"], dataset["access"], on="encoded_ip")
# delete ipv6 addr
# acc_with_ip = dataset[(dataset["isp"] != "")]
print("Loading 10%")
# acc_with_ip merge flint
acc_with_flint = pd.merge(
    dataset["access"], dataset["flint"], left_on="fqdn_no", right_on="fqdn_no_x"
)
print("Loading 40%")
# acc_with_flint merge flint
acc_with_fqdn = pd.merge(acc_with_flint, dataset["fqdn"], on="fqdn_no")
print("Loading 60%")
# acc_with_fqdn merge label & columns
columns_to_drop = ["fqdn_no", "fqdn_no_x", "date"]
acc_with_label = pd.merge(acc_with_fqdn, dataset["label"], on="fqdn_no").drop(
    columns=columns_to_drop, axis=1
)
print("Loading 80%")

current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
csv_filename = ".csv"
acc_with_label.to_csv(formatted_time + csv_filename)
print("Mission completed")
