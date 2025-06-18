import time
import pybullet as p
import pybullet_data
import creature
from environment import load_sandbox


def main():
    pid = p.connect(p.GUI)
    load_sandbox(pid)

    # generate a random creature and load it
    cr = creature.Creature(gene_count=3)
    xml_file = 'test.urdf'
    with open(xml_file, 'w') as f:
        f.write(cr.to_xml())
    cid = p.loadURDF(xml_file, (0, 0, 10), physicsClientId=pid)

    p.setRealTimeSimulation(1)
    while True:
        time.sleep(0.1)


if __name__ == '__main__':
    main()
