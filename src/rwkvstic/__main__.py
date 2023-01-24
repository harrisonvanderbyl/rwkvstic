from sys import argv


if __name__ == '__main__':
    import rwkvstic.preQuantize as pq
    if (argv[-1] == "--pq"):
        pq.preQuantized()
