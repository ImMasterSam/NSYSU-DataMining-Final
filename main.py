import sys
from testA import testA_main
from testB import testB_main

if __name__ == "__main__":

    if len(sys.argv) >= 2 and ('-l' in sys.argv[1:]):
        sys.stdout = open('output_A.log', 'w+', encoding='utf-8')

    # Run testA_main and testB_main
    testA_main()
    testB_main()