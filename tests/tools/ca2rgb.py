
def ca2rgb(inputPath, outputPath, palettePath):
    with open(palettePath, "rb") as paletteFile:
        palette = paletteFile.read()

    def r(i): return palette[i * 4 + 0].to_bytes(1, byteorder='little')
    def g(i): return palette[i * 4 + 1].to_bytes(1, byteorder='little')
    def b(i): return palette[i * 4 + 2].to_bytes(1, byteorder='little')

    with open(inputPath, "rb") as inputFile:
        inputBytes = inputFile.read()

        with open(outputPath, "wb") as outputFile:
            outputFile.write(inputBytes[:16])

            for byte in inputBytes[16:]:
                outputFile.write(r(byte))
                outputFile.write(g(byte))
                outputFile.write(b(byte))


if __name__ == '__main__':
    inputPath = "../data/dumps_01052019/MILL3N1I/palette-normal-16/facet_0_frame_1.dat"
    outputPath = "../data/facet_0_frame_1.rgb"
    palettePath = "../data/dumps_01052019/palette.dat"
    ca2rgb(inputPath, outputPath, palettePath)