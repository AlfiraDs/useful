import argparse



# print(parser.parse_args(['-a', '-bval', '-c', '3']))

# def main():
#     print(parser.parse_args())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Short sample app')

    parser.add_argument("--aaa", '-a', action="store_true", default=False)
    parser.add_argument("--bbb", '-b', action="store", dest="b")
    parser.add_argument("--ccc", '-c', action="store", dest="c", type=int)
    # parser.parse_args(['-a', '-bval', '-c', '3'])
    args = parser.parse_args(['-a', '-b', 'val', '-c', '3'])
    print(args.aaa)
    print(args)
