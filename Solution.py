import math
from collections import deque

class Solution:
    def main(self):
        n = 7168
        print(f'{n}:')
        print(self.numSquares(n))
        
    def numSquares(self, n: int) -> int:
        # Generate the list of perfect squares less than or equal to n
        squares = [i**2 for i in range(1, int(n**0.5) + 1)]
        
        # Use Breadth-First Search (BFS) with a queue
        queue = deque([n])
        depth = 0
        visited = set()
        
        while queue:
            depth += 1
            for _ in range(len(queue)):
                current = queue.popleft()
                for square in squares:
                    next_val = current - square
                    if next_val == 0:
                        return depth
                    if next_val < 0:
                        break
                    if next_val not in visited:
                        visited.add(next_val)
                        queue.append(next_val)
        
        return depth

if __name__ == '__main__':
    Solution().main()
