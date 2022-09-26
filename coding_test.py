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
#def longest_palindrome_substring(s):

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
# def three_sum(nums):
#     nums.sort()
#     for i in range(nums):



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
# def best_time_stock(prices):
#
#     return profit
#
# print(best_time_stock([7,1,5,3,6,4]))
# print(best_time_stock([7,6,4,3,1]))

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


l1 = ListNode(1, ListNode(2, ListNode(4)))
l2 = ListNode(1, ListNode(3, ListNode(4)))


print(merge_two_list(l1, l2).val)

def reverse_linked_list(head):
    rev = ListNode(next=head)

    while head:
