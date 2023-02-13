from sys import argv

if __name__ == '__main__':
    import rwkvstic.preQuantize as pq
    import rwkvstic.preJax as pj
    args = {
        "chunksize": 32 if "--cs" not in argv else int(argv[-1].split("=")[1]),
        "useLogFix": "--logfix" in argv,
    }
    pq.preQuantized(**args)
    if (argv[-1] == "--preJax"):
        pj.preJax()
