import os
import csv

def is_ds(dir):
	ds = ".DS_Store"
	if dir == ds: 
		return True
	return False

root_dir = "material_data"

def create_csv():
	file = open("labels.csv", "w")
	writer = csv.writer(file)
	for label_dir in os.listdir(root_dir):
		if is_ds(label_dir):
			continue
		for img in os.listdir(os.path.join(root_dir, label_dir)):
			if is_ds(img):
				continue
			img_dir = os.path.join(root_dir, label_dir, img)
			if label_dir == "paper":
				writer.writerow([img_dir, "1", "0", "0", "0", "0"])
			elif label_dir == "plastics":
				writer.writerow([img_dir, "0", "1", "0", "0", "0"])
			elif label_dir == "cans":
				writer.writerow([img_dir, "0", "0", "1", "0", "0"])
			elif label_dir == "glass":
				writer.writerow([img_dir, "0", "0", "0", "1", "0"])
			else:
				writer.writerow([img_dir, "0", "0", "0", "0", "1"])


def main():
	create_csv()

if __name__ == '__main__':
	main()

