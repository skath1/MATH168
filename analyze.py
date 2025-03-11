import zstandard as zstd

# Define file paths
zst_file = "/Users/shubhkathuria/Downloads/lichess_db_standard_rated_2016-02.pgn.zst"  # Replace this with your file path
pgn_file = "lichess_elite_2022-03.pgn"

# Decompress .zst file to .pgn
def extract_pgn(zst_file, output_pgn):
    dctx = zstd.ZstdDecompressor()
    with open(zst_file, 'rb') as ifh, open(output_pgn, 'wb') as ofh:
        dctx.copy_stream(ifh, ofh)

extract_pgn(zst_file, pgn_file)
print("Decompression complete. PGN file saved as:", pgn_file)
