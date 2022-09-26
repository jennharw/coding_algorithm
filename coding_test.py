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
            return [i, nums_map[target]]
        nums_map[num] = i