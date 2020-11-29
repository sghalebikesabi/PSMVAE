
def does_scroll(i):
    i_list = []
    save = i
    while save > 0:
        digit = save % 10
        i_list = [digit] + i_list
        save = save // 10
    visited = []
    pos = 0
    digit =  i_list[pos]
    while digit not in visited:
        visited += [digit]
        pos =  (digit + pos) % len(i_list)
        digit =  i_list[pos]
    
    return len(visited) == len(i_list)

for i in range(1, 10):
    print(i)

def find_all_scrolling_numbers(AB=[100, 500]):
    A = int(AB[0])
    B = int(AB[1])
    all_sn = []
    for i in range(A, B):
        if does_scroll(i):
            all_sn += str(i)
    if len(all_sn) == 0:
        all_sn = [str(-1)]
        
    all_sn = '\n'.join(all_sn)
    return(all_sn)

find_all_scrolling_numbers()