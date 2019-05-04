from PIL import Image


def png2rgb(inputPath, outputPath):
    img = Image.open(inputPath)

    rgb = bytearray()
    rgb.extend(img.width.to_bytes(4, 'little'))
    rgb.extend(img.height.to_bytes(4, 'little'))
    rgb.extend((img.width // 2).to_bytes(4, 'little'))
    rgb.extend((img.height // 2).to_bytes(4, 'little'))
    rgb.extend(img.tobytes())

    with open(outputPath, "wb") as outputFile:
        outputFile.write(rgb)


if __name__ == '__main__':
    inputPath = "../data/output.png"
    outputPath = "../data/output_back_to.rgb"
    png2rgb(inputPath, outputPath)
