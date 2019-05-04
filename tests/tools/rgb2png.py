from PIL import Image


def rgb2png(inputPath, outputPath):
    with open(inputPath, "rb") as inputFile:
        inputBytes = inputFile.read()

        width = int.from_bytes(inputBytes[0:4], 'little')
        height = int.from_bytes(inputBytes[4:8], 'little')

        img = Image.frombytes('RGB', (width, height), inputBytes[16:])
        img.save(outputPath)

if __name__ == '__main__':
    inputPath = "../data/input_bytes/frame1.rgb"
    outputPath = "../data/input_bytes/frame1.png"
    rgb2png(inputPath, outputPath)
