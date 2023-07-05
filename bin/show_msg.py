#Show a beautiful message box.
#example:
######################
#                    #
#    Hello world!    #
#                    #
######################
def show_msg(string):
    box_width = len(string) + 8
    for i in range(box_width):
        print("#", end = '')
    print()
    print("#", end = '')
    for i in range(box_width - 2):
        print(" ", end = '')
    print("#", end = '')
    print()
    print("#  ", string, "  #")
    print("#", end = '')
    for i in range(box_width - 2):
        print(" ", end = '')
    print("#", end = '')
    print()
    for i in range(box_width):
        print("#", end = '')
    print()
