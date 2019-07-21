def test(file):
    ids = train_loader.dataset.class_to_idx

    img = transform(file).unsqueeze(0)
    print(img)
    with torch.no_grad():
        out = model(img.to(device)).cpu().numpy()
        for key, value in ids.items():
            if value == np.argmax(out):
                # name = classes[int(key)]
                print(f"Predicted Label:and Key {key} and value {value}")
        plt.imshow(np.array(f))
        plt.show()
