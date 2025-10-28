import torch.nn as nn


class SnoutNet(nn.Module):
	def __init__(self):
		super().__init__()

		# --- Convolutional feature extractor ---
		self.features = nn.Sequential(
			# Conv1: i = 227x227x3 -> 227x227x64
			nn.Conv2d(3, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=4, ceil_mode=True),  # 227x227x64 -> 57x57x64

			# Conv2: i = 57x57x64 -> 57x57x128
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=4, ceil_mode=True),  # 57x57x128 -> 15x15x128

			# Conv3: i = 15x15x128 -> 15x15x256
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=4),  # 15x15x256 -> 4x4x256
		)

		# --- Fully connected regressor for (u, v) ---
		self.classifier = nn.Sequential(
			nn.Flatten(),  # 4x4x256 = 4096
			nn.Linear(256 * 4 * 4, 1024),  # FC1
			nn.ReLU(inplace=True),  # ReLU after FC1
			nn.Linear(1024, 1024),  # FC2
			nn.ReLU(inplace=True),  # ReLU after FC2
			nn.Linear(1024, 2)  # FC3 (u, v)
		)

	def forward(self, x):
		x = self.features(x)
		x = self.classifier(x)
		return x
