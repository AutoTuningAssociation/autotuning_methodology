from pathlib import Path

from kernel_tuner.cache.cli_tools import convert_t4

basepath = Path(__file__).parent
directories = ["convolution_milo", "dedisp_milo", "gemm_milo", "hotspot_milo"]

for directory in directories:
    print(f"Converting files in {directory}")
    dirpath = Path(basepath / directory)
    assert dirpath.is_dir(), f"Not a directory: {dirpath}"
    for infile in dirpath.iterdir():
        if infile.suffix.endswith("json") and not infile.stem.endswith("_T4"):
            print(f"  | converting {infile.stem}")
            outfile = infile.with_stem(infile.stem + "_T4")
            convert_t4(infile, outfile)
