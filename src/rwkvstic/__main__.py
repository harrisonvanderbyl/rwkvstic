from sys import argv

if __name__ == '__main__':
    import rwkvstic.preQuantize as pq
    import rwkvstic.preJax as pj
    if (argv[-1] == "--pq"):
        pq.preQuantized()
    if (argv[-1] == "--preJax"):
        pj.preJax()
