print("git test")
#코딩테스트 알고리즘 by 최하림

#파이썬 알고리즘 인터뷰
#6장 문자열 조작

#유효한 팰린드롬
import re
import collections
import sys
def valid_palindrome(s):
    s = s.lower()
    s = re.sub('[^a-z0-9]','', s) #substitue 치환
    return s == s[::-1]

print(valid_palindrome("race a car"))
print(valid_palindrome("A man, a plan, a canal: Panama"))

#문자열 뒤집기
def reverse_string(s:list[str]):
    #s.reverse()
    return s[::-1]

print(reverse_string(["h", "e", "l", "l", "o"]))
print(reverse_string(["H", "a", "n", "n", "a", "h"]))

#로그파일 재정렬
def reorder_logfiles(logs):
    digits = []
    letters = []
    for log in logs:
        if log.split()[1].isdigit():
            digits.append(log)
        else:
            letters.append(log)
    return sorted(letters, key = lambda x : (x.split()[1:], x.split()[0])) + digits

logs = ["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"]
print(reorder_logfiles(logs))
logs = ["a1 9 2 3 1","g1 act car","zo4 4 7","ab1 off key dog","a8 act zoo"]
print(reorder_logfiles(logs))

#가장 흔한 단어
def most_common_word(paragraph, banned):
    s = paragraph.lower()
    s = re.sub('[^a-z0-9]', ' ', s)
    counter = collections.Counter(s.split())
    for char, _ in counter.most_common():
        if char not in banned:
            return char

print(most_common_word(paragraph = "Bob hit a ball, the hit BALL flew far after it was hit.", banned = ["hit"]))
print(most_common_word(paragraph = "a.", banned = []))

#그룹 애너그램
def group_anagram(strs):
    dic = collections.defaultdict(list)
    for s in strs:
        dic[''.join(sorted(s))].append(s)
    return list(dic.values())

print(group_anagram(["eat","tea","tan","ate","nat","bat"]))
print(group_anagram([""]))
print(group_anagram(["a"]))

#가장 긴 팰린드롬 문자
def longest_palindrome_substring(s):
    def expand(left, right):
        while left> 0 and right < len(s) - 1 and s[left:right] == s[left:right][::-1] :
            left -= 1
            right += 1
        if s[left:right] == s[left:right][::-1]:
            print(s[left:right])
            return s[left:right]
        return ""

    result = ""
    for i in range(len(s)):
        result = max(result,
                     expand(i, i+1),
                     expand(i, i+2),
                     key=len)
    return result

print(longest_palindrome_substring("babad"))
print(longest_palindrome_substring("cbbd"))

#7장 배열
#두 수의 합
def two_sum(nums, target):

    nums_map = dict()
    for i, num in enumerate(nums):
        if target - num in nums_map:
            return [nums_map[target-num],i]
        nums_map[num] = i

print(two_sum(nums = [2,7,11,15], target = 9))
print(two_sum(nums = [3,2,4], target = 6))

#빗물 트래핑
def trapping_rain_water(height):
    left, right = 0, len(height) - 1
    left_max, right_max = height[left], height[right]
    result = 0
    while left <= right:
        if left_max <= right_max:
            left_max = max(left_max, height[left])
            result += left_max - height[left]
            left += 1
        else:
            right_max = max(right_max, height[right])
            result += right_max - height[right]
            right -= 1
    return result

print(trapping_rain_water([0,1,0,2,1,0,1,3,2,1,2,1]))
print(trapping_rain_water([4,2,0,3,2,5]))

#세 수의 합
def three_sum(nums):
    nums.sort()
    results = []
    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        p = nums[i]

        target = nums[i + 1:]
        left, right = 0, len(target) - 1
        while left < right:
            if p + target[left] + target[right] == 0:
                results.append([p, target[left], target[right]])
                left += 1
                right -= 1
            elif p + target[left] + target[right] > 0:
                right -= 1
            else:
                left += 1

    return results

print(three_sum([-1,0,1,2,-1,-4]))
print(three_sum([0,1,1]))
print(three_sum([0,0,0]))

#배열 파티션
def array_partition(array):
    array.sort()
    result = 0
    for i in range(0,len(array),2):
        result += array[i]

#배열파티션
def arrayPairSum(nums) -> int:
    sum = 0
    nums.sort()

    for i, n in enumerate(nums):
        # 짝수 번째 값의 합 계산
        if i % 2 == 0:
            sum += n

    return sum #sum(sorted(nums)[::2])

#자신을 제외한 곱
def product_of_array(nums):
    l, r = [], []
    tmp = 1
    for num in nums:
        l.append(tmp)
        tmp *= num
    tmp = 1
    for num in reversed(nums):
        r.append(tmp)
        tmp *= num
    results = []
    for i in range(len(nums)):
        results.append(l[i] * r[len(nums)-1-i])
    return results

print(product_of_array([1,2,3,4]))
print(product_of_array([-1,1,0,-3,3]))

#주식을 사고 팔기 좋은 시점
def best_time_stock(prices):
    profit = 0
    min_price = sys.maxsize

    for price in prices:
        min_price = min(min_price, price)
        profit = max(profit, price - min_price)

    return profit

print(best_time_stock([7,1,5,3,6,4]))
print(best_time_stock([7,6,4,3,1]))

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def palidrome_linked_list(head):
    slow, fast = head, head
    rev = ListNode()

    while fast and fast.next:
        fast = fast.next.next
        rev, slow, rev.next = slow, slow.next, rev

    if fast:
        slow = slow.next

    #rev, slow
    while slow:
        if slow.val != rev.val:
            return False
        slow, rev = slow.next, rev.next

    return True

head = ListNode(1, ListNode(2, ListNode(2, ListNode(1, None))))
print(palidrome_linked_list(head))
head = ListNode(1, ListNode(2, None))
print(palidrome_linked_list(head))

def merge_two_sorted_list(l1, l2):
    if (not l1) or (l2 and l1.val > l2.val):
        l1, l2 = l2, l1
    if l1:
        l1.next = merge_two_sorted_list(l1.next, l2)

    return l1 or l2

    return result
    return sum(sorted(array)[::2])

l1 = ListNode(1, ListNode(2, ListNode(4)))
l2 = ListNode(1, ListNode(3, ListNode(4)))

print(merge_two_sorted_list(l1, l2).val)

def reverse_linked_list(head):
    rev = ListNode(next=head)

    while head:
        rev, rev.next, head = head, rev, head.next
    return rev

head = ListNode(1, ListNode(2, ListNode(3,ListNode(4, ListNode(5)))))
print(reverse_linked_list(head))

def addTwoNumbers(l1, l2):
    root = rev = ListNode()

    carry, r = 0, 0
    while l1 and l2:
        carry, value = divmod(l1.val + l2.val + carry, 10)

        rev.next = ListNode(value)
        rev = rev.next
        l1, l2 = l1.next, l2.next
    if l2:
        rev.next = l2
    if l1:
        rev.next = l1
    return root.next
l1 = ListNode(2, ListNode(4, ListNode(3)))
l2 = ListNode(5, ListNode(6, ListNode(4)))
print(addTwoNumbers(l1, l2))

def swapNodesInPairs(head): # 12 34
    root = ListNode(head)
    prev, curr = root, head

    while prev and curr:
        tmp = curr.next.next
        prev.next = curr.next
        curr.next.next = curr
        curr.next = tmp

        prev, curr = curr, curr.next

    return root.next

head = ListNode(1, ListNode(2, ListNode(3, ListNode(4))))
print(swapNodesInPairs(head))


def oddEvenLinkedList(head):
    odd, even = ListNode(), ListNode()
    root = odd

    even_list = even

    while head:
        odd.next = head
        even.next = head.next
        odd = odd.next
        if even.next:
          even = even.next
        if head.next:
          head = head.next.next
        else:
          head = head.next

    odd.next = even_list.next
    return root.next
head = ListNode(1, ListNode(2, ListNode(3,ListNode(4, ListNode(5)))))
print(oddEvenLinkedList(head).next.next.next.next.val)

def reverseBetween(head, left, right):
    prev, curr = head, head.next
    for _ in range(left - 2):
        prev, curr = prev.next, curr.next

    far_left = curr

    root = ListNode(next=curr)
    rev = root

    for _ in range(right - left + 1):
        rev, rev.next, curr = curr, rev, curr.next

    prev.next = rev
    far_left.next = curr

    return head


head = ListNode(1, ListNode(2, ListNode(3,ListNode(4, ListNode(5)))))
print(reverseBetween(head, 2, 4))

#9장 스택, 큐
#유효한 괄호
def valid_parentheses(s):
    dic = {
        ")" : "(",
        "]" : "[",
        "}" : "{"
    }
    stack = []
    for char in s:
        if char in dic.keys():
            if stack.pop() != dic[char]:
                return False
        else:
            stack.append(char)

    if stack:
       return False
    return True

print(valid_parentheses(s = "()[]{}"))
print(valid_parentheses(s = "()[]{}"))
print(valid_parentheses(s = "(]"))

def remove_duplicate_letters(s):
    #중복된 문자 제거, 최대한 lexicographical order
    counter = collections.Counter(s)
    result = []
    for char in s:
        while result  and char <= result[-1] and counter[result[-1]] > 0:
            result.pop()
        if char not in result:
            result.append(char)
        counter[char] -= 1
    return ''.join(result)

print(remove_duplicate_letters("bcabc"))
print(remove_duplicate_letters("cbacdcbc"))

def daily_temperatures(temperatures):
    result = [0] * len(temperatures)
    stack = []

    for i, t in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < t:
            result[stack[-1]] = i - stack[-1]
            stack.pop()
        stack.append(i)
    return result

print(daily_temperatures([73,74,75,71,69,72,76,73]))
print(daily_temperatures([30,40,50,60]))
print(daily_temperatures([30,60,90]))

#큐를 이용한 스택 구현, implements stack using queue
class MyStack:
    def __init__(self):
        self.queue = collections.deque()

    def push(self, x):
        self.queue.append(x)
        for _ in range(len(self.queue)-1):
            self.queue.append(self.queue.popleft())

    def top(self):
        return self.queue[0]

    def pop(self):
        return self.queue.popleft()

    def empty(self):
        return len(self.queue) == 0

#스택을 이용한 큐 규현 implement queue using stacks
class MyQueue:
    def __init__(self):
        self.input = []
        self.output = []

    def push(self, x):
        self.input.append(x)

    def pop(self):
        self.peek()
        return self.output.pop()

    def peek(self):
        if not self.output:
            while self.input:
                self.output.append(self.input.pop())

        return self.output[-1]

    def empty(self):
        return self.input == [] and self.output == []

class MyCircularQueue:
    def __init__(self, size):
        self.q = [None] * size
        self.maxlen = size
        self.p1 = 0
        self.p2 = 0

    def enQueue(self, value):

        if self.q[self.p2] is None:
            self.q[self.p2] = value
            self.p2 = (self.p2+1) % self.maxlen
            return True
        else:
            return False

    def deQueue(self):
        if self.q[self.p1] is None:
            return False
        else:
            self.q[self.p1] = none
            self.p1 = (self.p1 + 1) % self.maxlen
            return True

    def Front(self):
        return -1 if self.q[self.p1] is None else self.q[self.p1]

    def Rear(self):
        return -1 if self.q[self.p2 - 1] is None else self.q[self.p2 - 1]

    def isEmpty(self):
        return self.p1 == self.p2 and self.q[self.p1] is None

    def isFull(self):
        return self.p1 == self.p2 and self.q[self.p1] is not None





#10장 Deque, Heapq
#k개 정렬 리스트 병합

# class ListNode:
#     def __init__(self, val, left, right):
#         self.val = val
#         self.left = left
#         self.right = right

class MyCircularDeque:
    def __init__(self, k):
        self.head, self.tail = ListNode(None), ListNode(None)
        self.k, self.len = k, 0
        self.head.right, self.tail.left = self.tail, self.head

    def __add(self, node, new):
        n = node.right
        node.right = new
        new.left, new.right = node, n
        n.left = new

    def insertFront(self, value):
        if self.len == self.k:
            return False
        self.len += 1
        self._add(self.head, ListNode(value))
        return True

    def insertLast(self, value):
        if self.len == self.k:
            return False
        self.len += 1
        self._add(self.tail.left, ListNode(value))
        return True

    def _del(self, node):
        n = node.right.right
        node.right = n
        n.left = node

    def deleteFront(self):
        if self.len == 0:
            return False
        self.len -= 1
        self._del(head)
        return True

    def deleteLast(self):
        if self.len == 0:
            return False
        self.len -= 1
        self._del(self.tail.left.left)
        return True

    def getFront(self):
        return self.head.right.val if self.len else -1

    def getRear(self):
        return self.tail.left.val if self.len else -1

    def isEmpty(self):
        return self.len == 0

    def isFull(self):
        return self.len == self.k

import heapq
def merge_k_sorted_lists(lists):
    heap = []
    i = 0
    for l in lists:
        heapq.heappush(heap, (l.val, i, l))
        i += 1

    root = rev = ListNode()
    while heap:
        val, i, l = heapq.heappop(heap)
        rev.next = l
        rev = rev.next
        if l.next:
            heapq.heappush(heap, (l.next.val, i, l.next))
    return root.next

l1 = ListNode(1, ListNode(4, ListNode(5)))
l2 = ListNode(1, ListNode(3, ListNode(4)))
l3 = ListNode(2, ListNode(6))
lists = [l1, l2, l3]
print(merge_k_sorted_lists(lists).next.next.next.next.val)

#11장 해시테이블
#
# class ListNode:
#     def __init__(self, key, val):
#         self.key = key
#         self.val = val
#         self.next = None

class MyHashMap:
    def __init__(self):
        #개별 체이닝
        self.size = 100
        self.table = collections.defaultdict(ListNode)

    def put(self, key, value):
        index = key % self.size
        if self.table[index].value is None:
            self.table[index] = ListNode(key, value)
            return
        p = self.table[index]
        while p:
            if p.key == key:
                p.val = value
                return
            if p.next is None:
                break
            p = p.next
        p.next = ListNode(key, value)


    def get(self, key):
        index = key % self.size
        if self.table[index].value is None:
            return -1

        p = self.table[index]
        while p:
            if p.key == key:
                return p.value
            p = p.next
        return -1

    def remove(self, key):
        index = key % self.size
        if self.table[index] is None:
            return

        p = self.table[index]
        if p.key == key:
            self.table[index] = ListNode() if p.next is None else p.next
            return

        prev = p
        while p:
            if p.key == key:
                prev.next = p.next
                return
            prev, p = p, p.next


#보석과 돌
def jewels_and_stones(jewels, stones):
    counter = collections.Counter(stones)
    result = 0
    for j in jewels:
        result += counter[j]

    return result

print(jewels_and_stones(jewels = "aA", stones = "aAAbbbb"))
print(jewels_and_stones(jewels = "z", stones = "ZZ"))

def longest_substring(s): #without repeating characters
    left, right = 0, 0
    counter = collections.Counter(s)
    seen = set()
    max_len = 0
    for char in s:
        while char in seen:
            seen.remove(s[left])
            left += 1

        right += 1
        seen.add(char)
        max_len = max(max_len, right - left)

    return max_len

print(longest_substring("abcabcbb"))
print(longest_substring("bbbbb"))
print(longest_substring("pwwkew"))

def top_k_frequent_elements(nums, k):
    counter = collections.Counter(nums)
    results = []
    result = counter.most_common(k)
    for r, c in result:
        results.append(r)
    return results

print(top_k_frequent_elements(nums = [1,1,1,2,2,3], k = 2))
print(top_k_frequent_elements(nums = [1], k = 1))

#4부 비선형 자료구조
#12장 그래프
def number_of_islands(grid):
    visited = [[False] * len(grid[0]) for _ in range(len(grid))]

    def dfs(x, y):
        dx = [-1,1,0,0]
        dy = [0,0,-1,1]

        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]

            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == '1' and visited[nx][ny] ==False:
                visited[nx][ny] = True
                dfs(nx,ny)
    result = 0
    for p in range(len(grid)):
        for q in range(len(grid[0])):
            if visited[p][q] == False and grid[p][q] == '1':
                dfs(p, q)
                result += 1
    return result
grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
print(number_of_islands(grid))

grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
print(number_of_islands(grid))

def letter_combinations_of_phone_number(digits):
    phone = {
        "2" : "abc",
        "3" : "def",
        "4" : "ghi",
        "5" : "jkl",
        "6" : "mno",
        "7" : "pqrs",
        "8" : "tuv",
        "9" : "wxyz"
    }
    result = []

    def dfs(digit, i):
        if len(digit) == 0:
            result.append(i)
            return

        for char in phone[digit[0]]:
            dfs(digit[1:], i + char)
    dfs(digits, "")

    return result

print(letter_combinations_of_phone_number("23"))
print(letter_combinations_of_phone_number(""))
print(letter_combinations_of_phone_number("2"))

import itertools
def permutation(nums):
    results = []
    prev_elements = []

    def dfs(elements):
      print(elements, prev_elements)
      if len(elements) == 0:
          results.append(prev_elements[:])
          return

      for e in elements:
          new_elements = elements[:]
          new_elements.remove(e)

          prev_elements.append(e)
          dfs(new_elements)
          prev_elements.remove(e)
          new_elements.append(e)

    dfs(nums)
    return results
    # return list(itertools.permutations(nums))

print(permutation([1,2,3]))

def combinations(n, k):
    # nums = [i + 1 for i in range(n)]
    # print(list(itertools.combinations(nums, k)))

    results = []

    def dfs(nums, p, k):
        if k == 0:
            results.append(nums[:])
            return
        for i in range(p, n+1):
            nums.append(i)
            dfs(nums, i+1, k-1)
            nums.remove(i)

    dfs([], 1, k)

    return results

print(combinations(4, 2))
print(combinations(1, 1))

def combination_sum(candidates, target):
    results = []
    def dfs(nums, i, target):
        if target == 0:
            results.append(nums[:])
            return
        if target < 0:
            return


        for c in range(i, len(candidates)):
            nums.append(candidates[c])
            dfs(nums, c, target - candidates[c])
            nums.remove(candidates[c])

    dfs([],0, target)
    return results

print(combination_sum([2,3,6,7], 7))
print(combination_sum([2,3,5], 8))

def subsets(nums):
    results = []

    def dfs(s, i):
        results.append(s[:])

        if i == len(nums):
            return

        for n in range(i, len(nums)):
            s.append(nums[n])
            dfs(s, n+1)
            s.remove(nums[n])

    dfs([], 0)

    return results

print(subsets([1,2,3]))
print(subsets([0]))

def reconstruct_itinerary(tickets):
    graph = collections.defaultdict(list)
    for f, t in tickets:
        graph[f].append(t)
        graph[f].sort()
    result = []
    def dfs(to):
        result.append(to)
        for v in graph[to]:
            graph[to].pop(0)
            dfs(v)

    dfs('JFK')
    return result

print(reconstruct_itinerary(tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]))
print(reconstruct_itinerary(tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]))

def course_schedule(numCourses, prerequistes):
    #cycle 이 존재하는지
    graph = collections.defaultdict(list)
    for v, c in prerequistes:
        graph[v].append(c)

    visited = []

    def dfs(v):
        if v in visited:
            return False

        visited.append(v)

        for c in graph[v]:
            if not dfs(c):
                return False
        visited.append(c)
        return True

    for x in list(graph):
        if not dfs(x):
            return False
    return True

#13장 최단경로
def network_delay_time(times, n, k):
    graph = collections.defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))

    Q = [(0, k)]
    dist = collections.defaultdict(int)

    while Q:
        time, node = heapq.heappop(Q)
        if node not in dist:
            dist[node] = time
            for v, w in graph[node]:
                alt = time + w
                heapq.heappush(Q, (alt, v))

    if len(dist) == n:
        return max(dist.values())
    return -1

print(network_delay_time(times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2))
print(network_delay_time(times = [[1,2,1]], n = 2, k = 1))
print(network_delay_time(times = [[1,2,1]], n = 2, k = 2))

def cheapest_flights_within(n, flights, src, dst, k):
    graph = collections.defaultdict(list)
    for x, y, w in flights:
        graph[x].append((y, w))

    Q = [(0, src, k)]

    while Q:
        t, v, _k = heapq.heappop(Q)
        if v == dst:
            return t
        if _k >= 0:
            for neighbor, m in graph[v]:
                heapq.heappush(Q, (t +m, neighbor,_k - 1))

    return -1

print(cheapest_flights_within(n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], src = 0, dst = 3, k = 1))
print(cheapest_flights_within(n = 3, flights = [[0,1,100],[1,2,100],[0,2,500]], src = 0, dst = 2, k = 1))
print(cheapest_flights_within(n = 3, flights = [[0,1,100],[1,2,100],[0,2,500]], src = 0, dst = 2, k = 0))


#14장 트리
#이진트리의 최대 깊이
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def maximum_depth_of_BT(root):
    if root is None:
        return 0
    left = maximum_depth_of_BT(root.left)
    right = maximum_depth_of_BT(root.right)
    return max(left, right) + 1

root = TreeNode(val= 3, left= TreeNode(val= 9, left= None, right= None), right= TreeNode(val= 20, left= TreeNode(val= 15, left= None, right= None), right=TreeNode(val= 7, left= None, right= None)))
print(maximum_depth_of_BT(root))
root = TreeNode(val= 1, left= None, right= TreeNode(val= 2))
print(maximum_depth_of_BT(root))

class Solution:
    def __init__(self):
        self.longest = 0
    def diameter_of_BT(self, root):

        def diameter(root):
            if root is None:
                return -1

            left = diameter(root.left)
            right = diameter(root.right)

            self.longest = max(left+right+2, self.longest)

            return max(left, right) + 1
        diameter(root)
        return self.longest


root = TreeNode(val=1, left=TreeNode(val=2, left=None, right=None),
                right=TreeNode(val=3, left=TreeNode(val=4, left=None, right=None),
                               right=TreeNode(val=5, left=None, right=None)))
S = Solution()
print(S.diameter_of_BT(root))

root = TreeNode(val = 1, left = TreeNode(val = 2))
S = Solution()
print(S.diameter_of_BT(root))


def longest_univalue_path(root):
    longest = [0]
    def univalalue(root):
        if root is None:
            return 0

        left = univalalue(root.left)
        right = univalalue(root.right)
        if root.left and root.right and root.left.val == root.val == root.right.val:
            longest[0] = max(longest[0] , left + right + 2)
            return left+right+2
        elif root.right and root.right.val == root.val:
            longest[0]  = max(longest[0] , right + 1)
            return right + 1
        elif root.left and root.left.val == root.val:
            longest[0]  = max(longest[0] , left + 1)
            return left + 1
        else:
            return 0

    univalalue(root)
    return longest[0]

root = TreeNode(val=5, left=TreeNode(val=4, left=TreeNode(val=1), right=TreeNode(val=1)),
                right=TreeNode(val=5, left=None,
                               right=TreeNode(val=5, left=None, right=None)))

print(longest_univalue_path(root))


root = TreeNode(val=1, left=TreeNode(val=4, left=TreeNode(val=4), right=TreeNode(val=4)),
                right=TreeNode(val=5, left=None,
                               right=TreeNode(val=5, left=None, right=None)))

print(longest_univalue_path(root))

def invert_binary_tree(root):
    if root is None:
        return None
    root.left, root.right = invert_binary_tree(root.right), invert_binary_tree(root.left)

    return root

root = TreeNode(val= 4, left= TreeNode(val= 2, left= TreeNode(val= 1, left= None, right= None), right= TreeNode(val= 3, left= None, right= None)), right= TreeNode(val= 7, left= TreeNode(val= 6, left= None, right= None), right= TreeNode(val= 9, left= None, right= None)))

print(invert_binary_tree(root).right.right.val)

root = TreeNode(val= 2, left= TreeNode(val= 1), right = TreeNode(val = 3))
print(invert_binary_tree(root).left.val)

def merge_two_binary_trees(root1, root2):
    root = TreeNode()

    if root1 and root2:
        root.val = root1.val + root2.val
    elif root1:
        root.val = root1.val
    elif root2:
        root.val = root2.val
    else:
        return None

    if root1.left and root2.left:
        root.left = merge_two_binary_trees(root1.left, root2.left)
    elif root1.left:
        root.left = root1.left
    elif root2.left:
        root.left = root2.left
    else:
        root.left = None

    if root1.right and root2.right:
        root.right = merge_two_binary_trees(root1.right, root2.right)
    elif root1.right:
        root.right = root1.right
    elif root2.right:
        root.right = root2.right
    else:
        root.right = None

    return root


root1 = TreeNode(val= 1, left= TreeNode(val= 3, left= TreeNode(val= 5, left= None, right= None)), right= TreeNode(val= 2))

root2 = TreeNode(val= 2, left= TreeNode(val= 1, right= TreeNode(val= 4, left= None, right= None)), right= TreeNode(val= 3, right= TreeNode(val= 7, left= None, right= None)))

print(merge_two_binary_trees(root1, root2).left.right.val)


#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        queue = collections.deque([root])
        result = ['#']
        while queue:
            node = queue.popleft()

            if node:
                queue.append(node.left)
                queue.append(node.right)
                result.append(node.val)

            else:
                result.append('#')

        return ' '.join(result)

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        if data == '# #':
            return None
        nodes = data.split()

        root = TreeNode(int(nodes[1]))
        queue = collections.deque([root])
        index = 2

        while queue:
            node = queue.popleft()
            if nodes[index] is not '#':
                node.left = TreeNode(int(nodes[index]))
                queue.append(node.left)
            index += 1
            if nods[index] is not "#":
                node.right = TreeNode(int(nodes[index]))
                queue.append(node.right)
            index += 1
        return root

# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))

#균형 이진 트리
def balanced_binary_tree(root):
    res = [0]
    def height(root):
        if root is None:
            return -1
        left = height(root.left)
        right = height(root.right)

        res[0] = max(abs(left - right), res[0])

        return max(left, right) + 1

    height(root)
    if res[0] <= 1:
        return True
    else:
        return False

root = TreeNode(val= 3, left= TreeNode(val= 5), right= TreeNode(val= 20, left=TreeNode(val=15), right = TreeNode(val=7)))
print(balanced_binary_tree(root))

root = TreeNode(val= 1, left= TreeNode(val= 2, left = TreeNode(val=3, left = TreeNode(val=4), right = TreeNode(val=4)), right=TreeNode(val=3)), right= TreeNode(val= 2))
print(balanced_binary_tree(root))

#최소높이 트리
def minimum_height_trees(n, edges):
    graph = collections.defaultdict(list)
    leaves = []
    for x, y in edges:
        graph[x].append(y)
        graph[y].append(x)

    for g in graph:
        if len(graph[g]) == 1:
            leaves.append(g)

    while n > 2:
        n -= len(leaves)
        new_leaves = []
        for leaf in leaves:
            for neighbor in graph[leaf]:
                graph[neighbor].remove(leaf)
                if len(graph[neighbor]) == 1:
                    new_leaves.append(neighbor)
        leaves = new_leaves

    return leaves

print(minimum_height_trees(4, [[1,0],[1,2],[1,3]]))
print(minimum_height_trees( n = 6, edges = [[3,0],[3,1],[3,2],[3,4],[5,4]]))

#정렬된 배열의 이진 탐색트리 변환
def sorted_array_to_bst(nums):
    if nums == []:
        return None
    index =  len(nums) // 2
    root = TreeNode(nums[index])
    root.left = sorted_array_to_bst(nums[:index])
    root.right = sorted_array_to_bst(nums[index+1:])

    return root

print(sorted_array_to_bst([-10,-3,0,5,9]).right.val)
print(sorted_array_to_bst([1,3]).left.val)

#이진 탁색 트리(BST)를 더 큰 수 합계 트리로
def bst_to_greater_sum_tree(root):
    res = [0]
    def greater(root):
        if root:
            greater(root.right)
            res[0] += root.val
            root.val = res[0]
            greater(root.left)
        return root
    greater(root)
    return root

root = TreeNode(val= 4, left= TreeNode(val= 1, left= TreeNode(val=0), right = TreeNode(val=2, left = None, right =TreeNode(val = 3))), right= TreeNode(val= 6, left=TreeNode(val=5), right = TreeNode(val=7, right=TreeNode(8))))
print(bst_to_greater_sum_tree(root).right.val)

#이진 탐색 트리(BST)합의 범위
def range_sum_of_bst(root, low, high):
    res = [0]
    def sum(root):
        if root is None:
            return 0
        if low <= root.val <= high:
            res[0] += root.val
            sum(root.right)
            sum(root.left)
        elif root.val < high:
            sum(root.right)
        else:
            sum(root.left)

    sum(root)
    return res[0]

root = TreeNode(val= 10, left= TreeNode(val= 5, left=TreeNode(3), right=TreeNode(7)), right= TreeNode(val= 15, right = TreeNode(val=18)))
print(range_sum_of_bst(root, 7, 15))
root = TreeNode(val= 10, left= TreeNode(val= 5, left=TreeNode(3, left=TreeNode(1)), right=TreeNode(7, left=TreeNode(6))), right= TreeNode(val= 15, left = TreeNode(val=13),right = TreeNode(val=18)))
print(range_sum_of_bst(root, 6, 10))


#이진탐색트리 노드간 최소 거리
def minimum_distance_nodes(root):
    INF = (1e9)
    res = [INF]

    def minimum_d(root):
        if root is None:
            return
        if root.right and root.left:
            res[0] = min(res[0], root.right.val - root.val, root.val - root.left.val)
            minimum_d(root.right)
            minimum_d(root.left)
        elif root.right:
            res[0] = min(res[0], root.right.val - root.val)
            minimum_d(root.right)
        elif root.left:
            res[0] = min(res[0], root.val - root.left.val)
            minimum_d(root.left)
        else:
            return

    minimum_d(root)
    return res[0]
root = TreeNode(val= 4, left= TreeNode(val= 2, left=TreeNode(1), right=TreeNode(3)), right= TreeNode(val= 6))
print(minimum_distance_nodes(root))
root = TreeNode(val= 2, left= TreeNode(val= 0,), right= TreeNode(val= 48, left=TreeNode(12), right= TreeNode(50)))
print(minimum_distance_nodes(root))

#전위, 중위 순회 결과로 이진 트리 구축
def buildTree(preorder, inorder):
    if inorder:
        index = inorder.index(preorder.pop(0))

        root = TreeNode(inorder[index])
        root.left = buildTree(preorder, inorder[:index])
        root.right = buildTree(preorder, inorder[index+1:])
        return root

print(buildTree(preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]).val)
print(buildTree(preorder = [-1], inorder = [-1]).val)


#15장 힙
class BinaryHeap:
    def __init__(self):
        self.items = [None]

    def __len__(self):
        return len(self.items) - 1

    def _percolate_up(self, i):
        parent = i // 2
        if parent > 0:
            if self.items[i] < self.items[parent]:
                self.items[parent], self.items[i] = self.items[i], self.items[parent]
            self._percolate_up(parent)

    def insert(self, k):
        self.items.append(k)
        self._percolate_up(len(self))

    def _percolate_down(self, i):
        left = i * 2
        right = i * 2 + 1
        smallest = i
        if left <= len(self) and self.items[smallest] > self.items[left]:
            smallest = left

        if right <= len(self) and self.items[smallest] > self.items[right]:
            smallest = right

        if smallest  != i:
            self.items[smallest], self.items[i] = self.items[i], self.items[smallest]
            self._percolate_up(smallest)

    def extract(self):
        extracted = self.items[1]

        self.items[1] = self.items[len(self)]
        self.items.pop()
        self._percolate_down(1)

        return extracted


BH = BinaryHeap()
BH.insert(5)
BH.insert(9)
BH.insert(18)
BH.insert(7)
BH.insert(3)
print(BH.items)
BH.extract()
print(BH.items)
BH.extract()
print(BH.items)

def kth_largest_element(nums, k):
    Q = []
    for i, n in enumerate(nums):
        heapq.heappush(Q, (-n, i))

    for _ in range(k-1):
        heapq.heappop(Q)

    return -heapq.heappop(Q)[0]

print(kth_largest_element([3,2,3,1,2,4,5,5,6], 4))

#15장 트라이

#트라이 구현
# class TrieNode:
#     def __init__(self):
#         self.word = False
#         self.children = collections.defaultdict(TrieNode)
#
# class Trie:
#     def __init__(self):
#         self.root = TrieNode()
#
#     def insert(self, word):
#         node = self.root
#         for char in word:
#             node = node.children[char]
#         node.word = True
#
#     def search(self, word):
#         node = self.root
#         for char in word:
#             if char not in node.children:
#                 return False
#             node = node.children[char]
#         return node.word
#
#     def startsWith(self, prefix):
#         node = self.root
#         for char in prefix:
#             if char not in node.children:
#                 return False
#             node = node.children[char]
#         return True
#
# trie = Trie()
# trie.insert("apple")
# print(trie.search("apple"))
# print(trie.search("app"))
# print(trie.startsWith("app"))
# trie.insert("app")
# print(trie.search("app"))

#펠린드롬 페어

class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.word_ids = -1
        self.palindrome_ids = []

class Trie:
    def __init__(self):
        self.root = TrieNode()

    @staticmethod
    def is_palindrome(word):
        return word == word[::-1]

    def insert(self, idx, word):
        node = self.root
        for i, w in enumerate(reversed(word)):
            if self.is_palindrome(word[:len(word)-i]):
                node.palindrome_ids.append(idx)
            node = node.children[w]
        node.word_ids = idx

    def search(self, idx, word):
        result = []
        node = self.root
        for i, char in enumerate(word):
            # 3 단어가 남았을 경우
            if node.word_ids != -1 and self.is_palindrome(word[i:]):
                result.append([idx, node.word_ids])

            if char not in node.children:
                return result
            node = node.children[char]

        #2
        for x in node.palindrome_ids:
            result.append([idx, x])

        # 1 단어 abcd dcba
        if node.word_ids != idx and node.word_ids != -1:
            result.append([idx, node.word_ids])

        return result

def palindrome_pairs(words):
    results = []

    trie = Trie()
    for i, word in enumerate(words):
        trie.insert(i, word)

    for i, word in enumerate(words):
        results.extend(trie.search(i, word))
    return results


print(palindrome_pairs(words = ["abcd","dcba","lls","s","sssll"]))
print(palindrome_pairs(words = ["bat","tab","cat"]))
print(palindrome_pairs(words =["a",""]))