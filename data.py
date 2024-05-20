def __getitem__(self, idx):
    img_path = os.path.join(self.images_dir, self.files[idx])
    clean = Image.open(img_path).convert('RGB')

    # Convert clean to a NumPy array
    clean = np.array(clean)

    # random crop
    i = np.random.randint(clean.shape[0] - self.image_size[0])
    j = np.random.randint(clean.shape[1] - self.image_size[1])

    # Crop the image using NumPy array slicing
    clean = clean[i:i+self.image_size[0], j:j+self.image_size[1]]

    # Convert NumPy array to PyTorch tensor
    clean = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0

    # Normalize the tensor
    transform = tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    clean = transform(clean)

    # Add noise
    noisy = clean + 2 / 255 * self.sigma * torch.randn(clean.shape)

    return noisy, clean
