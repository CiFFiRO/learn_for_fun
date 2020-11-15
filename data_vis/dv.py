import graphviz


def plot_ex_1():
    tree = graphviz.Graph()
    limit = 16
    queue = [0]
    tree.node('0', '0')
    number = 0
    while len(queue) > 0:
        v = queue.pop(0)
        if v+2 > limit:
            break
        l = number + 1
        r = number + 2
        tree.node(str(l), str(l))
        tree.node(str(r), str(r))
        tree.edge(str(v), str(l))
        tree.edge(str(v), str(r))
        queue.append(l)
        queue.append(r)
        number += 2
    tree.render('ex_1.gv', view=True)


def plot_ex_2():
    proc = graphviz.Digraph()
    proc.node('C', 'Создание')
    proc.node('Rd', 'Готов к выполнению')
    proc.node('R', 'Выполняется')
    proc.node('B', 'Блокирован')
    proc.node('Z', 'Зомби')
    proc.edge('C', 'Rd')
    proc.edge('Rd', 'R')
    proc.edge('R', 'B')
    proc.edge('B', 'Rd')
    proc.edge('R', 'Z')
    proc.render('ex_2.gv', view=True)


if __name__ == '__main__':
    plot_ex_2()


