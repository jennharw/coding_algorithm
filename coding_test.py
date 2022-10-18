import copy
import datetime

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
            self.q[self.p1] = None
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
            if node[index] is not "#":
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

#17장 정렬

#버블정렬
#버블정렬
def bubblesort(array):
  for _ in range(len(array)):
    for j in range(len(array)-1):
      if array[j] > array[j+1]:
         array[j], array[j+1] = array[j+1], array[j]

  return array

array = [4,7,9,5,3,2,1,6,8,10]
print(bubblesort(array))

#병합정렬
def mergesort(array):
  if len(array) == 1:
    return array
  left = mergesort(array[:len(array)//2])
  right = mergesort(array[len(array)//2:])

  result = []
  i, j = 0, 0
  while i < len(left) and j < len(right):
    if left[i] <= right[j]:
      result.append(left[i])
      i += 1
    else:
      result.append(right[j])
      j += 1
  if i == len(left):
    result.extend(right[j:])
  if j == len(right):
    result.extend(left[i:])
  return result

array = [4,7,9,5,3,2,1,6,8,10]
print(mergesort(array))

#퀵정렬
def quicksort(array, start, end):
    if start >= end:
        return

    pivot = start
    left = start + 1
    right = end

    while left <= right:
        # left
        while left <= end and array[left] <= array[pivot]:
            left += 1
        while right > start and array[right] >= array[pivot]:
            right -= 1

        if left > right:
            array[pivot], array[right] = array[right], array[pivot]
        else:
            array[left], array[right] = array[right], array[left]

    quicksort(array, start, right - 1)
    quicksort(array, right + 1, end)
    return array

array = [4,7,9,5,3,2,1,6,8,10]
quicksort(array, 0, len(array) - 1)

#선택정렬
#선택정렬
def selectionsort(array):
	for i in range(len(array)):
		min_index = i
		for j in range(i, len(array)):
			if array[j] < array[min_index]:
				min_index = j
		array[i], array[min_index] = array[min_index], array[i]

	return array

array = [5, 7, 9, 0, 3, 1, 6, 2, 4, 8]

print(selectionsort(array))


# 삽입정렬

def insertionsort(array):
    for i in range(1, len(array)):
        for j in range(i, 0, -1):  # 인덱스 i부터 1까지 1씩 감소하며 반복하는 문법
            if array[j] < array[j - 1]:  # 한 칸씩 왼쪽으로 이동
                array[j], array[j - 1] = array[j - 1], array[j]
            else:  # 자기보다 작은 데이터를 만나면 그 위치에서 멈춤
                break
    return array


array = [5, 7, 9, 0, 3, 1, 6, 2, 4, 8]
print(insertionsort(array))

#계수 정렬
def counting_sort(lst):
    mina=min(lst)
    maxa = max(lst)

    counting = [0] * len(lst)
    for i in range(len(lst)):
        counting[lst[i]-mina]+=1 #counting

    #printing
    result = []
    for i in range(len(counting)):
        for j in range(counting[i]):
            result.append(i+mina)

    return result

import math
def radix_sort(array):
    D = int(math.log10(max(array)))
    print("D", D)
    for i in range(D + 1):
        bucket = []
        for j in range(0, 10):
            bucket.append([])
        for j in range(len(array)):
            digit = int(array[j] // math.pow(10, i)) % 10
            bucket[digit].append(array[j])

        cnt = 0
        for j in range(0, 10):
            for i in bucket[j]:
                array[cnt] = i
                cnt += 1

    return array


array = [508, 73, 29, 6, 3, 12, 6, 19, 71, 7]
print(radix_sort(array))

def merge(left, right):
    a = b = 0

    cArr = []

    while a < len(left) and b < len(right):
        if left[a] < right[b]:
            cArr.append(left[a])
            a += 1
        elif left[a] > right[b]:
            cArr.append(right[b])
            b += 1
        else:
            cArr.append(left[a])
            cArr.append(right[b])
            a += 1
            b += 1

    #if a < len(left):
    cArr.extend(left[a:])

    #if b < len(left):
    cArr.extend(right[b:])
    return cArr


#삽입정렬
def insertion_sort(l, left=0, right=None):
    if right is None:
        right = len(l)-1
    for i in range(left+1, right+1):
        key = l[i]
        #j = binary_search(l, key, left+1, right+1)
        j = i -1
        while j >= left and l[j] > key:
            l[j+1] = l[j]
            j -= 1 #binary 아님

        l[j+1] = key
    return

def tim_sort(l):
    min_run = 4
    n = len(l)
    for i in range(0, n, min_run):
        insertion_sort(l, i, min((i+min_run-1), (n-1)))
    #min_run단위로 sort

    size = min_run

    while size<n:
        for s in range(0, n, size*2):
            mid = s + size - 1
            end = int(min((s + size*2 -1), (n - 1)))
            merged = merge(left = l[s:mid+1], right= l[mid+1: end+1])
            l[s:s+len(merged)] = merged
        size *= 2
    return l

array = [5, 7, 9, 6, 3, 1, 6, 9,7, 7,88]
print(tim_sort(array))

#리스트 정렬
def merge_list(l1, l2):
    if l1 and l2:
        if l1.val > l2.val:
          l1, l2 = l2, l1
        l1.next = merge_list(l1.next, l2)
    return l1 or l2

def sort_list(head):
    if not (head and head.next):
        return head

    half, slow, fast = None, head, head

    while fast and fast.next :
        half, slow, fast = slow, slow.next, fast.next.next
    half.next = None

    left = sort_list(head)
    right =  sort_list(slow)

    return merge_list(left, right)

head = ListNode(4, ListNode(2, ListNode(1, ListNode(3, None))))
print(sort_list(head).val)
head = ListNode(-1, ListNode(5, ListNode(3, ListNode(4, ListNode(0)))))
print(sort_list(head).val)


def merge_intervals(intervals):
    stack = [intervals[0]]

    for i in range(1, len(intervals)):
        if stack[-1][1] > intervals[i][0]:
            stack[-1][1] = max(stack[-1][1], intervals[i][1])
        else:
            stack.append(intervals[i])

    return stack


print(merge_intervals(intervals=[[1, 3], [2, 6], [8, 10], [15, 18]]))

def insertion_sort_list(head):
    dummy = ListNode(0, head)

    prev, cur = head, head.next

    while cur:
        if prev.val <= cur.val:
            prev, cur = cur, cur.next
            continue

        temp = dummy
        while temp.next.val < cur.val:
            temp = temp.next

        prev.next = cur.next
        cur.next = temp.next
        temp.next = cur

        cur = prev.next

    return dummy.next

head = ListNode(4, ListNode(2, ListNode(1, ListNode(3))))
print(insertion_sort_list(head).val)
head = ListNode(-1, ListNode(5, ListNode(3, ListNode(4, ListNode(0)))))
print(insertion_sort_list(head).val)


def calculate(a, b):
	a = str(a)
	b = str(b)
	return  int(a+b) < int(b+a)

print(calculate(10, 2))
def largest_number(nums):
  i = 1
  while i < len(nums):
    j = i
    while j > 0 and calculate(nums[j-1], nums[j]):
      nums[j], nums[j-1] = nums[j-1], nums[j]
      j -= 1
    i += 1

  return ''.join(list(map(str, [2,10])))

print(largest_number([10,2]))
print(largest_number( [3,30,34,5,9]))

def valid_anagram(s, t):
  return sorted(s) == sorted(t)

print(valid_anagram(s = "anagram",t = "nagaram"))
print(valid_anagram(s = "rat", t = "car"))

def sort_color(nums):
  red, white, blue = 0, 0, len(nums)-1
  while white < blue:
    if nums[white] > 1:
      nums[blue], nums[white] = nums[white], nums[blue]
      blue -= 1
    elif nums[red] < 1:
      nums[white], nums[red] = nums[red] , nums[white]
      white += 1
      red += 1
    else:
      white += 1
  return nums
print(sort_color([2,0,2,1,1,0]))
print(sort_color([2,0,1]))

import heapq
import math
def Kclosest(points, k):
  Q = []
  for x, y in points:
    heapq.heappush(Q, (math.log(x**2 + y ** 2) , [x, y]))

  result = []
  for _ in range(k):
    d, point = heapq.heappop(Q)
    result.append(point)
  return result
print(Kclosest(points = [[1,3],[-2,2]], k = 1))
print(Kclosest(points = [[3,3],[5,-1],[-2,4]], k = 2))

#18장 이진 검색
import bisect
def binary_search(nums, target):

  for i in nums:
     r = bisect.bisect_left(nums, i)
     if nums[r] == target:
       return r
  return -1

#이진 검색
def binary_search(nums, target):
  left, right = 0, len(nums) - 1

  while left <= right:
    mid = (left + right) // 2
    if nums[mid] == target:
      return mid
    if nums[mid] < target:
      left = mid + 1
    else:
      right = mid - 1

  return -1

print(binary_search(nums = [-1,0,3,5,9,12], target = 9))
print(binary_search(nums = [-1,0,3,5,9,12], target = 2))
print(binary_search(nums = [-1,0,3,5,9,12], target = 9))
print(binary_search(nums = [-1,0,3,5,9,12], target = 2))

#회전 정렬된 배열 검색
def search(nums, target) -> int:
  left, right = 0, len(nums) - 1

  while left <= right:
    mid = (left + right) // 2
    if nums[mid] == target:
      return mid

    if nums[mid] < target:
      if nums[mid] < nums[left]:
          left = mid + 1
      else:
          right = mid - 1
    else:
      if nums[mid] > nums[right]:
        left = mid + 1
      else:
        right = mid - 1

  return -1


print(search(nums = [4,5,6,7,0,1,2], target = 0))
print(search(nums = [4,5,6,7,0,1,2], target = 3))
print(search(nums = [1], target = 0))

#두 배열의 교집합
def intersection(nums1, nums2):
    nums2.sort()
    result = set()
    for n in nums1:
        i = bisect.bisect_left(nums2, n)
        if nums2[i] == n:
            result.add(n)
    return result

print(intersection(nums1 = [1,2,2,1], nums2 = [2,2]))
print(intersection(nums1 = [4,9,5], nums2 = [9,4,9,8,4]))

def intersection(nums1, nums2):
  nums2.sort()
  nums1.sort()
  result = set()
  i = j = 0
  while i < len(nums1) and j < len(nums2):
    if nums1[i] > nums2[j]:
      j += 1
    elif nums1[i] < nums2[2]:
      i += 1
    else:
      result.add(nums1[i])
      i += 1
      j += 1

  return result

print(intersection(nums1 = [1,2,2,1], nums2 = [2,2]))
print(intersection(nums1 = [4,9,5], nums2 = [9,4,9,8,4]))

#두 수의 합
def two_sum(numbers, target):
    for i in range(len(numbers)):
        k = bisect.bisect_left(numbers, target - numbers[i])
        if numbers[k] + numbers[i] == target:
            return [i + 1, k + 1]


print(two_sum(numbers=[2, 7, 11, 15], target=9))
print(two_sum(numbers = [2,3,4], target = 6))

#2D 행렬 검색
def searchMatrix(matrix, target):
    i = 0
    j = len(matrix[0]) - 1
    while i < len(matrix) and j >= 0:
        if matrix[i][j] == target:
            return True
        if matrix[i][j] > target:
            j -= 1
        else:
            i += 1
    return False

print(searchMatrix(
    matrix=[[1, 4, 7, 11, 15], [2, 5, 8, 12, 19], [3, 6, 9, 16, 22], [10, 13, 14, 17, 24], [18, 21, 23, 26, 30]],
    target=5))
print(searchMatrix(
    matrix=[[1, 4, 7, 11, 15], [2, 5, 8, 12, 19], [3, 6, 9, 16, 22], [10, 13, 14, 17, 24], [18, 21, 23, 26, 30]],
    target=20))

#19장 비트 조작

#싱글 넘버
def singleNumber(nums):
  i = nums[0]
  for k in nums[1:]:
    i ^= k
  return i

print(singleNumber([2,2,1]))
print(singleNumber([4,1,2,1,2]))

#해밍 거리
def hammingDistance(x, y):
  return bin(x ^ y).count('1')
print(hammingDistance(1, 4))
print(hammingDistance(1, 3))

#두 정수의 합
def two_sum_bits(a, b):

    while b != 0:
        c = a ^ b
        b = (a & b) << 1
        a = c

    return a

print(two_sum_bits(1,2))
print(two_sum_bits(2,3))
print(two_sum_bits(9,11))

#UTF-8 Validation
def validate_utf8(data):
  i = 0
  while i < len(data):
    digit_num= 0
    n = data[i]
    if  n > 255:

      return False
    elif  n & (128+64+32) == (128+64):
      digit_num  +=1
    elif  n & (128+64+32+16) == (128+64+32):
      digit_num  +=2
    elif  n & (128+64+32+16+8) == (128+64+32+16):
      digit_num  +=3
    elif n & 128 == 0 :
      digit_num = 0
    else:

      return False

    for _ in range(digit_num):
      i +=1
      if (data[i] & (128+64) != 128):
        return False
    i += 1
  return True

print(validate_utf8(data = [197,130,1]))
print(validate_utf8(data = [235,140,4]))

#1비트의 개수
def number_of_1bits(n):
    return bin(n).count('1')

print(number_of_1bits(n = "00000000000000000000000000001011"))
print(number_of_1bits(n = "00000000000000000000000010000000"))
print(number_of_1bits(n = "11111111111111111111111111111101"))


#20장 슬라이딩 윈도우
#최대 슬라이딩 윈도우
def maxSlidingWindow(nums, k):
    window = collections.deque()
    max_win = float('-inf')
    result = []
    for n in nums:
        window.append(n)
        if len(window) < k - 1:
            continue
        if max_win == float('-inf'):
            max_win = max(window)
        elif n > max_win:
            max_win = n

        result.append(max_win)

        if max_win == window.popleft():
            max_win = float('-inf')

    return result

print(maxSlidingWindow(nums=[1, 3, -1, -3, 5, 3, 6, 7], k=3))
print(maxSlidingWindow(nums = [1], k = 1))


#부분 문자열이 포함된 최소 윈도우
def minWindow(s, t):
    need = collections.Counter(t)
    missing = len(t)

    left = start = end = 0

    for right, char in enumerate(s, 1):

        missing -= need[char] > 0
        need[char] -= 1

        if missing == 0:
            while (need[s[left]] < 0 and left < right):
                need[s[left]] += 1
                left += 1

            if end == 0 or right - left < end - start:
                start, end = left, right
            need[s[left]] += 1
            left += 1
            missing += 1

    return s[start:end]


print(minWindow(s="ADOBECODEBANC", t="ABC"))
print(minWindow(s="a", t="a"))
print(minWindow(s="a", t="aa"))

def longestCharacter(s, k):
    left = right = 0
    counts = collections.Counter()
    for right in range(1, len(s) + 1):
        counts[s[right - 1]] += 1
        n = counts.most_common(1)[0][1]

        if right - left - n > k:
            counts[s[left]] -= 1
            left += 1

    return right - left


print(longestCharacter(s="ABAB", k=2))
print(longestCharacter(s="AABABBA", k=1))

#21장 그리디 알고리즘
#주식을 사고 팔기 좋은 시점
def profit(prices):
  result = 0

  for i in range(len(prices)-1):
    if prices[i+1] > prices[i]:
      result += prices[i+1] - prices[i]

  return result

print(profit(prices = [7,1,5,3,6,4]))
print(profit(prices = [1,2,3,4,5]))
print(profit(prices = [7,6,4,3,1]))

#키에따른 대기열 재구성
import heapq
def height(people):
  heap = []

  for p, i in people:
    heapq.heappush(heap, (-p, i))

  result = []

  while heap:
    p, i = heapq.heappop(heap)
    result.insert(i, [-p, i])
  return result

print(height([[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]))
print(height(people = [[6,0],[5,0],[4,0],[3,2],[2,2],[1,4]]))

#태스크 스케줄러
def leastIntervals(tasks, n):
  counter = collections.Counter(tasks)
  result = 0

  while True:
    sub_count = 0
    for task, _ in counter.most_common(n+1):
     sub_count += 1
     result += 1
     counter.subtract(task)

     counter +=  collections.Counter()


    if not counter:
       break

    result += n - sub_count + 1
  return result

print(leastIntervals(tasks = ["A","A","A","B","B","B"], n = 2))
print(leastIntervals(tasks = ["A","A","A","B","B","B"], n = 0))
print(leastIntervals(tasks = ["A","A","A","A","A","A","B","C","D","E","F","G"], n = 2))


#주요소
def gas(gas, cost):
    if sum(gas) < sum(cost):
        return -1
    result = 0
    answer = -1
    for i in range(len(gas)):

        result += gas[i] - cost[i]
        if result < 0:
            result = 0
            answer = i + 1

    return answer


print(gas(gas=[1, 2, 3, 4, 5], cost=[3, 4, 5, 1, 2]))
print(gas(gas=[2, 3, 4], cost=[3, 4, 3]))

#쿠키
def assign_cookies(g, s):
    g.sort()  # 아가
    s.sort()
    i = 0
    j = 0

    while i < len(g) and j < len(s):
        if g[i] <= s[j]:
            i += 1
            j += 1
        else:
            j += 1

    return i


print(assign_cookies(g=[1, 2, 3], s=[1, 1]))
print(assign_cookies(g=[1, 2], s=[1, 2, 3]))

#22장 분할 정복

#과반수 element
def majority_element(nums):
    nums.sort()
    return nums[len(nums) // 2]

print(majority_element(nums = [3,2,3]))
print(majority_element(nums = [2,2,1,1,1,2,2]))

#괄호를 사용하는 여러 방법
def different_ways_to_add_parentheses(expression):
    def compute(left, right, op):
        result = []
        for l in left:
            for r in right:
                result.append(eval(str(l) + op + str(r)))
        return result

    if expression.isdigit():
        return [int(expression)]
    results = []
    for index, ex in enumerate(expression):
        if ex in '+-*':
            left = different_ways_to_add_parentheses(expression[:index])
            right = different_ways_to_add_parentheses(expression[index+1:])
            results.extend(compute(left, right, ex))
    return results

print(different_ways_to_add_parentheses("2-1-1"))
print(different_ways_to_add_parentheses("2*3-4*5"))

#23장 다이나믹 프로그래밍

def fibonacci(n):
    if n == 0:
        return 0
    if n ==1:
        return 1
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(2))

import collections
def fibonacci_dp(n):
    dp = collections.defaultdict(int)
    dp[0] = 0
    dp[1] = 1
    for i in range(n-1):
        dp[i+2] = dp[i] + dp[i+1]

    return dp[n]

print(fibonacci_dp(3))
print(fibonacci_dp(4))

def maximum_subarray(nums):
    for i in range(1, len(nums)):
        nums[i] += nums[i-1] if nums[i-1] > 0 else 0
    return max(nums)

import sys
def maximum_subarray_dp(nums):
    best_sum = -sys.maxsize
    current_sum = 0

    for num in nums:
        current_sum = max(num, current_sum+num)
        best_sum = max(best_sum, current_sum)
    return best_sum

print(maximum_subarray_dp(nums = [-2,1,-3,4,-1,2,1,-5,4]))
print(maximum_subarray_dp(nums = [1]))
print(maximum_subarray_dp(nums = [5,4,-1,7,8]))

def climbing_stairs(n):
    dp = collections.defaultdict(int)
    dp[1] = 1
    dp[2] = 2
    for i in range(3, n+1):
        dp[n] = dp[n-1] + dp[n-2]
    return dp[n]

print(climbing_stairs(2))
print(climbing_stairs(3))

def house_robber(nums):
    dp = collections.defaultdict(int)
    dp[0] = 0
    dp[1] = nums[0]

    for i in range(2, len(nums)+1):
        dp[i] = max(dp[i-1], dp[i-2] + nums[i-1])
    return max(dp.values())

print(house_robber(nums = [1,2,3,1]))
print(house_robber(nums = [2,7,9,3,1]))

cargo = [
        (4, 12),
        (2, 1),
        (10, 4),
        (1, 1),
        (2, 2),
    ]
def fractional_knapsack_problem(cargo):
    capacity = 15
    pack = []

    for c in cargo:
        pack.append([c[0]/c[1], c[0], c[1]])
    pack.sort(reverse = True)

    total_value = 0
    for p in pack:
        if capacity - p[2] >= 0:
            capacity -= p[2]
            total_value += p[1]
        else:
            fraction = capacity/p[2]
            total_value += p[1] * fraction
            break
    return total_value

def zero_one_knapsack(cargo):
    capacity = 15
    pack = []

    for i in range(len(cargo)+1):
        pack.append([])
        for c in range(capacity + 1):
            if i == 0 or c == 0:
                pack[i].append(0)
            elif cargo[i-1][1] <= c:
                pack[i].append(
                    max(
                        pack[i-1][c],
                        cargo[i-1][0]+pack[i-1][c - cargo[i-1][1]]
                    )
                )

            else:
                pack[i].append(pack[i-1][c])
    return pack[-1][-1]

#카카오

#비밀지도
def solution1(n, arr1, arr2):
    result = []
    for i in range(n):
        result.append(bin(arr1[i] | arr2[i])[2:].zfill(n).replace('1', '#').replace('0',' '))
    return result


print(solution1(n =	5, arr1	= [9, 20, 28, 18, 11], arr2	= [30, 1, 21, 17, 28]))
print(solution1(n =6, arr1	 = [46, 33, 33 ,22, 31, 50], arr2	= [27 ,56, 19, 14, 14, 10]))

#다트게임
def solution2(dartResult):
    result = []
    s = ''
    for i in dartResult:
        if i == 'S':
            result.append(int(s) ** 1)
            s = ''
        elif i == 'D':
            result.append(int(s) ** 2)
            s = ''
        elif i == 'T':
            result.append(int(s) ** 3)
            s = ''
        elif i == '*':
            result[-1] = result[-1] * 2
            if len(result) > 1: result[-2] = result[-2] * 2
        elif i == '#':
            result[-1] = -result[-1]
        else:
            s += i
    return sum(result)

print(solution2("1S2D*3T"))
print(solution2("1D2S#10S"))
print(solution2("1D2S0T"))
print(solution2("1S*2T*3S"))
print(solution2("1D#2S*3S"))
print(solution2("1T2D3D#"))
print(solution2("1D2S3T*"))

import collections

#캐시
def solution3(cacheSize, cities):
    window = collections.deque(maxlen=cacheSize)
    result = 0
    for city in cities:
        city = city.lower()
        if city in window:
            #재삽입
            result += 1
            window.remove(city)
            window.append(city)
        else:
            result += 5
            window.append(city)

    return result


print(solution3(3, ["Jeju", "Pangyo", "Seoul", "NewYork", "LA", "Jeju", "Pangyo", "Seoul", "NewYork", "LA"]))
print(solution3(3, ["Jeju", "Pangyo", "Seoul", "Jeju", "Pangyo", "Seoul", "Jeju", "Pangyo", "Seoul"]))
print(solution3(2, ["Jeju", "Pangyo", "Seoul", "NewYork", "LA", "SanFrancisco", "Seoul", "Rome", "Paris", "Jeju",
                    "NewYork", "Rome"]))
print(solution3(5, ["Jeju", "Pangyo", "Seoul", "NewYork", "LA", "SanFrancisco", "Seoul", "Rome", "Paris", "Jeju",
                    "NewYork", "Rome"]))
print(solution3(2, ["Jeju", "Pangyo", "NewYork", "newyork"]))
print(solution3(0, ["Jeju", "Pangyo", "Seoul", "NewYork", "LA"]))

#셔틀버스
def solution4(n, t, m, timetable):
    timetable = [int(time[:2]) * 60 + int(time[3:])  for time in timetable]
    timetable.sort()
    current = 60* 9
    for _ in range(n):
        for _ in range(m):
            if timetable and timetable[0] <= current:
                candidate = timetable.pop(0) - 1
            else:
                candidate = current

        current += t
    h, m = divmod(candidate, 60)
    return str(h).zfill(2) + ":" + str(m).zfill(2)

print(solution4(1,1, 5, ["08:00", "08:01", "08:02", "08:03"]))
print(solution4(2, 10, 2, ["09:10", "09:09", "08:00"]))
print(solution4(2, 1, 2, 	["09:00", "09:00", "09:00", "09:00"]))
print(solution4(1, 1, 5, ["00:01", "00:01", "00:01", "00:01", "00:01"]))
print(solution4(1, 1, 1, 	["23:59"]))
print(solution4(10, 60, 45, 	["23:59","23:59", "23:59", "23:59", "23:59", "23:59", "23:59", "23:59", "23:59", "23:59", "23:59", "23:59", "23:59", "23:59", "23:59", "23:59"]))

import collections
import re


#뉴스클러스터링
def solution5(str1, str2):
    str1 = [str1[i:i + 2].lower() for i in range(len(str1) - 1) if re.findall('[a-z]{2}', str1[i:i + 2].lower())]
    str2 = [str2[i:i + 2].lower() for i in range(len(str2) - 1) if re.findall('[a-z]{2}', str2[i:i + 2].lower())]
    str1 = collections.Counter(str1)
    str2 = collections.Counter(str2)
    intersection = sum((str1 & str2).values())
    union = sum((str1 | str2).values())

    jarcard_sim = 1 if union == 0 else intersection / union
    return int(jarcard_sim * 65536)


print(solution5('FRANCE', 'french'))
print(solution5('handshake', 'shake hands'))
print(solution5('aa1+aa2', 'AAAA12'))
print(solution5('E=M*C^2	', 'e=m*c^2	'))

#프렌즈4 블록
def solution6(m, n, board):
    board = [list(x) for x in board]

    matched = True
    while matched:
        matched = []
        # 삭제될 것이 있는지 확인
        for i in range(m - 1):
            for j in range(n - 1):
                if board[i][j] == board[i + 1][j] == board[i][j + 1] == board[i + 1][j + 1] != '#':
                    matched.append((i, j))

        # 삭제
        for i, j in matched:
            board[i][j] = board[i + 1][j] = board[i][j + 1] = board[i + 1][j + 1] = '#'

        # 다시 내려오기
        for _ in range(m):
            for i in range(m):
                for j in range(n):
                    if i > 0 and board[i][j] == '#':
                        board[i][j], board[i - 1][j] = board[i - 1][j], '#'

    return sum(x.count('#') for x in board)


print(solution6(4, 5, ["CCBDE", "AAADE", "AAABF", "CCBBF"]))
print(solution6(6, 6, ["TTTANT", "RRFACC", "RRRFCC", "TRRRAA", "TTMMMF", "TMMTTJ"]))

#추석 트래픽import datetime
def solution7(lines):
    #시작, 끝
    packed = []
    for line in lines:
        logs = line.split(' ')
        timestamp = datetime.datetime.strptime(logs[0] + ' ' + logs[1], "%Y-%m-%d %H:%M:%S.%f").timestamp()

        timestamp = datetime.datetime.strptime(logs[0] + ' ' + logs[1], "%Y-%m-%d %H:%M:%S.%f").timestamp()
        packed.append((timestamp, -1))
        packed.append((timestamp - float(logs[2][:-1]) + 0.001, 1))

    accumulated = 0
    max_requests = 1
    packed.sort(key=lambda x: x[0])
    for i, elem1 in enumerate(packed):
        current = accumulated #종료되지 않은 누적된 request

        for elem2 in packed[i:]:
            if elem2[0] - elem1[0] > 0.999:
                break
            if elem2[1] > 0:
                current += elem2[1] #1초 내 시작되는 request
        max_requests = max(max_requests, current)
        accumulated += elem1[1]

    return max_requests

print(solution7([
"2016-09-15 01:00:04.001 2.0s",
"2016-09-15 01:00:07.000 2s"
]))
print(solution7([
"2016-09-15 01:00:04.002 2.0s",
"2016-09-15 01:00:07.000 2s"
]))
print(solution7([
"2016-09-15 20:59:57.421 0.351s",
"2016-09-15 20:59:58.233 1.181s",
"2016-09-15 20:59:58.299 0.8s",
"2016-09-15 20:59:58.688 1.041s",
"2016-09-15 20:59:59.591 1.412s",
"2016-09-15 21:00:00.464 1.466s",
"2016-09-15 21:00:00.741 1.581s",
"2016-09-15 21:00:00.748 2.31s",
"2016-09-15 21:00:00.966 0.381s",
"2016-09-15 21:00:02.066 2.62s"
]))

#이코테
#3장 그리디

#거스름돈
def changes(n):
    count = 0
    coin_types = [500, 100, 50, 10]

    for coin in coin_types:
        count += n // coin
        n %= coin
    return count

print(changes(1260))

#큰수의 법칙
# n = int(input())
# plans = input().split()
# n, m, k = map(int, input().split())
# data = list(map(int, input().split()))
def principle_of_big_number(n, m, k, data):
    data.sort()
    first = data[n-1]
    second = data[n-2]

    count = int(m / (k+1)) * k
    count += m % (k+1)

    result = 0
    result += count * first
    result += (m - count) * second
    return result

    # result = 0
    # while True:
    #     for i in range(k):
    #         if m == 0:
    #             break
    #         result += first
    #         m -= 1
    #     if m == 0:
    #         break
    #     result += second
    #     m -= 1
    return result

print(principle_of_big_number(5,8,3,[2,4,5,4,6]))

#숫자게임
def card_game(n, m, data):
    result = 0
    for i in range(n):
        min_value = min(data[i])
        result = max(result, min_value)
    return result

print(card_game(3, 3, [[3,1,2],[4,1,4],[2,2,2]]))
print(card_game(2, 4, [[7,3,1,8],[3,3,3,4]]))

#1이 될 때까지
def until_1(n, k):
    result = 0

    while True:
        target = (n // k) * k
        result += (n - target)

        n = target
        if n < k:
            break
        result += 1
        n //= k

    result += (n - 1)
    return result

print(until_1(25, 5))
print(until_1(17, 4))


#4장 구현
def LRUD(n, grid):
    #무시 공간 밖 시작 1,1
    dic = {'L':0,
           'R':1,
           'U':2,
           'D':3}
    dx = [-1,1,0,0]
    dy = [0,0,-1,1]
    x, y = 0, 0
    for g in grid:
        if 0 <= x + dx[dic[g]] < n and 0 <= y + dy[dic[g]] < n:
            x += dx[dic[g]]
            y += dy[dic[g]]

    return y+1, x+1

print(LRUD(5, 'RRRUDD'))

def time(n):
    coun = 0
    for i in range(n+1):
        for j in range(60):
            for k in range(60):
                if '3' in str(i) + str(j) + str(k):
                    count += 1
    return count

print(time(5))


def knight(start):
    steps = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, 2)]

    x = int(start[1]) - 1
    y = ord(start[0]) - 97

    count = 0
    for i in range(8):
        if 0 <= x + steps[i][0] < 8 and 0 <= y + steps[i][1] < 8:
            count += 1
    return count


print(knight('a1'))
print(knight('c2'))

def developing_games(n, m, start, grid):
    #visited = [[False] * n for _ in range(m)]
    visited = [list(i) for  i in grid]

    dict = {
        '0':(-1,0),
        '1':(0,-1),
        '2':(1,0),
        '3':(0,1)
    }

    x = start[0]
    y = start[1]
    d = start[2]
    count = 0
    result = 0
    while True:

        if visited[x + dict[d][0]][y + dict[d][1]] == False:
            x += dict[d][0]
            y += dict[d][1]
            visited[x + dict[d][0]][y + dict[d][1]] = 1
            count = 0
        else:
            d = (d + 1) % 4
            count += 1
        if count == 4:
            x = x - dict[d][0]
            y = y - dict[d][1]

            if x < 0 or x > n or y < 0 or y > m or  grid[x][y] == 1 :

                return result

        result += 1
    return result


#5장 DFS/BFS
def implement_dfs(graph):

    def dfs(graph, v, visited):
        visited[v] = True
        print(v, end=' ')
        for i in graph[v]:
            if not visited[i]:
                dfs(graph, i, visited)

    visited = [False] * 9
    dfs(graph, 1, visited)

print(implement_dfs(graph = [
  [],
  [2, 3, 8],
  [1, 7],
  [1, 4, 5],
  [3, 5],
  [3, 4],
  [7],
  [2, 6, 8],
  [1, 7]
]))

def implement_bfs(graph):
    def bfs(graph, v, visited):

        queue = collections.deque([start])
        visited[start] = True
        while queue:
            v = queue.popleft()
            print(v, end= ' ')
            for i in graph[v]:
                if not visited[i]:
                    queue.append(i)
                    visited[i] = True

    bfs(graph, 1, visited)

print(implement_bfs(graph = [
  [],
  [2, 3, 8],
  [1, 7],
  [1, 4, 5],
  [3, 5],
  [3, 4],
  [7],
  [2, 6, 8],
  [1, 7]
]))

#음료수 얼려먹기





# 음료수 얼려먹기
def freezing_drink(grid):  # 총 0그룹이 몇 개인지
    visited = [[False] * len(grid[0]) for _ in range(len(grid))]

    def dfs(x, y, visited):
        visited[x][y] = True
        dx = [0, 0, 1, -1]
        dy = [-1, 1, 0, 0]
        for i in range(4):
            if 0 <= x + dx[i] < len(grid) and 0 <= y + dy[i] < len(grid[0]):
                nx = x + dx[i]
                ny = y + dy[i]
                if visited[nx][ny] == False and grid[nx][ny] == 0:
                    dfs(nx, ny, visited)

    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if visited[i][j] == False and grid[i][j] == 0:
                dfs(i, j, visited)
                count += 1
    return count


print(freezing_drink([[0, 0, 1, 1, 0], [0, 0, 0, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]))

#미로 탈출
import collections


def maze(n, m, grid):
    # 최소 dp 괴물 0

    def bfs(x, y):
        queue = collections.deque()
        queue.append((x, y))
        dx = [-1, 1, 0, 0]
        dy = [0, 0, 1, -1]
        while queue:
            x, y = queue.popleft()
            for i in range(4):
                nx = x + dx[i]
                ny = y + dy[i]

                if nx < 0 or nx >= n or ny < 0 or ny >= m:
                    continue
                if grid[nx][ny] == 0:
                    continue
                if grid[nx][ny] == 1:
                    grid[nx][ny] = grid[x][y] + 1
                    queue.append((nx, ny))
        return grid[n - 1][m - 1]

    return bfs(0, 0)


print(maze(3, 3, [[1, 1, 0], [0, 1, 0], [0, 1, 1]]))
print(maze(5, 6, [[1, 0, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]))

#정렬
#위에서 아래로
def uptodown(array):
    return sorted(array, reverse=True)

print(uptodown([15, 27, 12]))

#성적이 낮은 학생
def grade(array):
    array.sort(key = lambda x:x[1])
    array = [i[0] for i in array]
    return array

print(grade([("홍길동", 95), ("이순신", 77)]))

#두 배열의 원소 교체
def exchange(n, k, array1, array2):
    array1.sort()
    array2.sort(reversed =True)
    for i in range(k):
        if array1[i] < array2[i]:
            array1[i], array2[i] = array1[i], array2[i]

        else:
            break
    return sum(array1)

#이진 탐색
#빠르게 입력 받기
# import sys
# input_data = sys.stdin.readline().rstrip()

#부품찾기
import bisect
def binary_search(array, target, start, end):
    while start <= end:
        mid = (start + end) // 2
        if array[mid] == target:
            return mid
        elif array[mid] > target:
            end = mid - 1
        else:
            start = mid + 1
    return None
def find(n, n_list, m, m_list):
    n_list.sort()
    m_list.sort()

    for k in m_list:
        i = bisect.bisect_left(k, array, 0, n-1)
        if i != None:
            print("Yes", end = ' ')
        else:
            print('No', end = ' ')

    return
    array = set(n_list)
    for k in m_list:
        if k in array:
            print("Yes", end=' ')
        else:
            print('No', end=' ')

print(find(5, [8,3,7,9,2], 3, [5,7,9]))
#떡볶이 떡
def tteokbokki(n, m, t):
    t.sort()
    start, end = 0, len(t) - 1
    while start<=end:
        mid = (start + end) - 1
        result = 0
        for i in t:
            result += i - mid > 0
        if result == m:
            return mid
        elif result > m:
            start = mid + 1
        else:
            end = mid - 1
    return mid

print(tteokbokki(4, 6, [19,15,10,17]))

#다이나믹 프로그래밍
#1로 만들기
#개미전사
#바닥공사
#효율적인 화폐구성

def make1(N):
    dp = [0] * (N+1)

    for i in range(2, N+1):

        dp[i] = dp[i-1] + 1
        if i % 2 == 0:
            dp[i] = min(dp[i], dp[i // 2] + 1)
        if i % 3 == 0:
            dp[i] = min(dp[i], dp[i // 3] + 1)

        if i % 5 == 0:
            dp[i] = min(dp[i], dp[i // 5] + 1)

    return dp[N+1]
#개미전사
def ant(array):
    dp = [0] * len(array+1)

    for i in range(len(array)+1):
        if i == 1:
            dp[i] = array[i-1]
            continue
        dp[i] = max(dp[i-1], dp[i-2] + array[i-1])
    return dp[len(array)]

#바닥공사
def floor(N):
    #1x2 2x1 2x2 - 바닥을 채우는 모든 경우의 수
    dp = [0] * (N+1)
    dp[1] = 1
    dp[2] = 3
    for i in range(3,N):
        dp[i] = dp[i-2] + 2*dp[i-1]
    return dp[N]

print(floor(3))# 5

#다익스트라
import heapq
def dijkstra(n, k, start, graph):
    grid = collections.defaultdict()
    for x, y, z in graph:
        grid[x].append((y, z))

    heap = [(0, start)]
    distance = [0] * n
    while heap:
        dist, v = heapq.heappop(heap)
        for y, z in graph[v]:
            distance[y] += z + dist
            heapq.heappush(heap, (z+dist, y))

    return distance


print(dijkstra(6, 11, 1, [[1,2,2], [1,3,5], [1,4,1],[2,3,3],[2,4,2],[3,2,3],[3,6,5],[4,3,3],[4,5,1],[5,3,1],[5,6,2]]))
#모든 노드로 가기 위한 최단 거리
def floydwarshalle(n, k, graph):
    grid = collections.defaultdict(list)
    for x, y, z in graph:
        grid[x].append((y, z))
    dist = [[10001] * (n+1) for _ in range(n+1)]

    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist


print(floydwarshalle(4, 7, [[1,2,4],[1,4,6],[2,1,3],[2,3,7],[3,1,5],[3,4,4],[4,3,2]]))

#미래도시
def future(n,m, graph, x, k):#a가 1번에서  k 번회사를 거쳐 x 번 회사로 가는 최소 이동 시간
    #floyed warsharl
    grid = [[10001]  * (n+1) for _ in range(n+1)]

    grap = collections.defaultdict(list)
    for x, y in graph:
        grap[x].append(y)

    for k in range(n+1):
        for i in range(n+1):
            for j in range(n+1):
                grap[i][j] = min(grap[i][j], grap[i][k] + grap[k][j])

    return graph[1][k] + graph[k][x]


print(future(5,7,[[1,2],[1,3],[1,4],[2,4],[3,4],[3,5],[4,5]],4,5))
print(future(4,2,[[1,3],[2,4]],3,4))


def telegram(n, m, c, graph):
    #c에서 보낸 베시지를 받는 도시의 총 개수와 총 걸리는 시간
    grap = collections.defaultdict(list)
    for x, y, z in graph:
        grap[x].append((y, z))
    distance = [-1] * (n+1)
    distance[c] = 0
    heap = [(0, c)]
    while heap:
        dist, v = heapq.heappop(grap)
        for w, d in grap[v]:
            distance[w] = dist + d
            heapq.heappush(heap, ((dist+d, w)))



print(telegram(3,2,1,[[1,2,4],[1,3,2]]))



#크루스칼 알고리즘
#서로소 집합 알고리즘
#위상 정렬
def find_parent(parent, x):
	if parent[x] != x:
		parent[x] = find_parent(parent, parent[x])
	return parent[x]

def union_parent(parent, a, b):
	a = find_parent(parent, a)
	b = find_parent(parnet, b)
	if a < b:
		parnet[b] = a

	else:
p		parent[b] = a

# 부모 테이블상에서, 부모를 자기 자신으로 초기화
for i in range(1, v + 1):
    parent[i] = i

# 간선을 비용순으로 정렬
edges.sort()

for edge in edges:
	cost, a, b = edge
	if find_parent(parent, a) != find_parent(parnet, b):
		union_parnet(parnet, a, b)
		result += cost
# 각 원소가 속한 집합 출력하기
print('각 원소가 속한 집합: ', end='')
for i in range(1, v + 1):
    print(find_parent(parent, i), end=' ')

print()

#위상정렬
진입차수확인
# 모든 노드에 대한 진입차수는 0으로 초기화
indegree = [0] * (v + 1)

for a, b in graph:
	graph[a].append(b)
	indegree[a] += 1

def topology_sort():
	result = []
	q = deque()
	for i in range(1, v+1):
		now = q.popleft()
		result.append(now)
		for i in graph[now]:
			indegree[i] -= 1
			if indegree[i] == 0:
				q.append(i)
	for i in result:
        print(i, end=' ')

topology_sort()

#팀결성
def teammaking(n, m, operations):
    parent = [0] * (n+1)
    for oper, a, b in operations:
        if oper == 0:
            union_parent(parent, a, b)
        else:
            if find_parent(parent=parent, a) == find_parent(parent, b):
                print("y")
            else:
                print("n")

#도시분할
def divide_city(v, e, edges):
    parent = [0] * (v + 1)  # 부모 테이블 초기화

    # 부모 테이블상에서, 부모를 자기 자신으로 초기화
    for i in range(1, v + 1):
        parent[i] = i

    # 모든 간선을 담을 리스트와, 최종 비용을 담을 변수
    edges = []
    result = 0

    # 간선을 비용순으로 정렬
    edges.sort()
    last = 0  # 최소 신장 트리에 포함되는 간선 중에서 가장 비용이 큰 간선

    for cost, a, b in edges:
        if find_parent(parent, a) != find_parent(parnet, b):
            union_parent(parent=parent, a, b)
            result += cost
            last = cost
    return result - last
#커리쿨럼
def curriculum(v, ): #topology sort
    indegree = [0] * (v+1)
    graph = collections.defaultdict(list)
    time = [0] * (v+1)
    for x in data:
        indegree[i] += 1

        graph[x].append(i)


    def topology():
        result = copy.deepcopy(time)
        q = deque()

        for i in range(1, v+1):
            if indegree[i] == 0:
                q.append(i)

        while q:
            now = q.popleft()
            for i in graph[now]:
                result[i] = max(result[i], result[now] + time[i])
                indegree[i] -= 1
                if indegree[i] == 0:
                    q.append(i)

            # 위상 정렬을 수행한 결과 출력
        for i in range(1, v + 1):
            print(result[i])