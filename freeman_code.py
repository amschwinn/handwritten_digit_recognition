
# input: image (binary representation in 2d array)
# output: freeman code
def freeman_code(img):
    # find leftmost activated pixel of top row
    rows = len(img)
    cols = len(img[0])
    for i in xrange(rows):
        for j in xrange(cols):
            if img[i][j] == 1:
                start_row = i
                start_col = j
                break
        if img[i][j] == 1:
            break

    # traverse contour until back to starting position, saving directions along the way
    start = True
    current_row = start_row
    current_col = start_col
    last_row = -1
    last_col = -1
    code = []
    while ((current_col != start_col or current_row != start_row) and not start and direction != 8) or start:
        start = False
        direction, new_current_row, new_current_col = next_direction(img, current_row, current_col, rows, cols, last_row, last_col)
        last_row = current_row
        last_col = current_col
        current_row = new_current_row
        current_col = new_current_col
        img[last_row][last_col] = 0
        code.append(direction)

    return code[:-1]


# input: image, current row and column, total rows and columns, previous row and column
# output: direction, new current row and column
def next_direction(img, current_row, current_col, rows, cols, last_row, last_col):
    # making sure not at a border of the image (to prevent accessing pixel outside of image)
    top = False
    bottom = False
    left = False
    right = False
    if current_row == 0:
        top = True
    elif current_row == rows - 1:
        bottom = True
    if current_col == 0:
        left = True
    elif current_col == cols - 1:
        right = True

    # loop through directions in order to determine next direction
    if not top:
        if img[current_row - 1][current_col] == 1 and ((current_row - 1) != last_row or current_col != last_col):
            return 0, current_row - 1, current_col
        if not right:
            if img[current_row - 1][current_col + 1] == 1 and ((current_row - 1) != last_row or (current_col + 1) != last_col):
                return 1, current_row - 1, current_col + 1
    if not right:
        if img[current_row][current_col + 1] == 1 and ((current_row) != last_row or (current_col + 1) != last_col):
            return 2, current_row, current_col + 1
        if not bottom:
            if img[current_row + 1][current_col + 1] == 1 and ((current_row + 1) != last_row or (current_col + 1) != last_col):
                return 3, current_row + 1, current_col + 1
    if not bottom:
        if img[current_row + 1][current_col] == 1 and ((current_row + 1) != last_row or current_col != last_col):
            return 4, current_row + 1, current_col
        if not left:
            if img[current_row + 1][current_col - 1] == 1 and ((current_row + 1) != last_row or (current_col - 1) != last_col):
                return 5, current_row + 1, current_col - 1
    if not left:
        if img[current_row][current_col - 1] == 1 and (current_row != last_row or (current_col - 1) != last_col):
            return 6, current_row, current_col - 1
        elif not top:
            if img[current_row - 1][current_col - 1] == 1 and ((current_row - 1) != last_row or (current_col - 1) != last_col):
                return 7, current_row - 1, current_col - 1

    return 8, -1, -1


test_image0 =  [[0,0,1,0,0],
                [0,1,0,1,0],
                [1,0,0,0,1],
                [0,1,0,1,0],
                [0,0,1,0,0]]

print freeman_code(test_image0)
