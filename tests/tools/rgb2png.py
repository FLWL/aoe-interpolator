from PIL import Image

#inputPath = "../data/facet_0_frame_0.rgb"
#outputPath = "../data/facet_0_frame_0.png"
inputPath = "../data/output.rgb"
outputPath = "../data/output.png"

with open(inputPath, "rb") as inputFile:
    inputBytes = inputFile.read()

    width = int.from_bytes(inputBytes[0:4], 'little')
    height = int.from_bytes(inputBytes[4:8], 'little')

    print("RGB width: " + str(width))
    print("RGB height: " + str(height))

    img = Image.frombytes('RGB', (width, height), inputBytes[16:])
    #img.show()
    img.save(outputPath)
