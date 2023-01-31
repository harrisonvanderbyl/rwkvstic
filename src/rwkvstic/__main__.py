from sys import argv

if __name__ == '__main__':
    import rwkvstic.preQuantize as pq
    import rwkvstic.preJax as pj
    if ("--pq" in argv):
        if ("--cs" in argv[-1]):
            pq.preQuantized(chunksize=int(argv[-1].split("=")[1]))
        else:
            pq.preQuantized(chunksize=32)
    if (argv[-1] == "--preJax"):
        pj.preJax()
