class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str: 
        self.word1 = word1
        self.word2 = word2

        C = [''.join(z) for z in zip(word1, word2)]
        strings = "".join(C)
        return strings

sol = Solution()
print(sol.mergeAlternately("abc", "pqr"))

class Tree:
    def inorderTraversal(self, root):
     
        self.root = root
        k = len(root)
        k = int(k/2)
        right = root[k:]
        left = root[:k]

        while k <= 0:
            return []

        new_list = []
        for value in left:
            try:
                new_list.append(int(value))
            except ValueError:
                continue

        for value2 in right:
            try:
                new_list.append(int(value2))
            except ValueError:
                continue


        return new_list
        
sol = Tree()
sol.inorderTraversal([1, "null", 2, 3])

def twoSum(nums, target: int):
    if len(nums) > 3: 
   
        k = int(len(nums)/2)
            
        left = nums[k:]
        right = nums[:k]
        
        N=[]
        Y=[]
        for i in range(len(left)):
            if left[i] in nums:
                if sum(left) == target:
                    N.append(i)
            else:
                return []
        
        for y in range(len(right)):
            if right[y] in nums:
                if sum(right) == target:
                    Y.append(y + 2)
            else:
                return []
        
        print(N)
        print(Y)

twoSum([2,7,11,15], 9)

def twoSumreplica(nums, target):
     
    index=[]
    for i, number in enumerate(nums):
        complementary = target - number
        if complementary in nums[i:]:
            index.append([i, nums.index(complementary)])
            break
    else:
        print("No solutions exist")
    
    newlist = [item for items in index for item in items]
    return newlist

twoSumreplica([2, 7, 11,15], 9)

def groupAnagrams(strs):
    if strs == [""]:
        return [[""]]
    elif strs == ["", ""]:
        return [["", ""]]

    alphabetical = sorted(strs)
    new_dict = {}
    for y in alphabetical:
        strings = y.split()
        for i in strings:
            new = ''.join(sorted(i))  
            if new in new_dict:
                new_dict[new].append(i) # key append value
            else:
                new_dict[new] = [i]
    return list(new_dict.values())
                
groupAnagrams(["eat","tea","tan","ate","nat","bat"])

def firstMissingPositive(nums) -> int:

    count = [i for i in range(50)]
    if len(nums) > len(count):
        return len(nums)
    elif len(nums) == len(count):
        return int(len(nums/2))
    else:
        new_count = int(len(count)/len(nums))
        if [s % 2 for s in count[new_count:]]:
            new_count = int(len(count[new_count:])/2)
    
    left_count = count[:new_count]
    right_count = count[new_count:]

    i = set.intersection(set(nums),set(left_count))
    y = set.intersection(set(nums),set(right_count))

        
    if len(i) > 0:
        n = len(left_count)
        b = nums
        hash = set(b)
        for i in range(1, n + 2):
            if i not in hash:
                return i
              
    
    elif len(y) > 0:
        n = len(right_count)
        b = nums
        hash = set(b)
        for i in range(1, n + 2):
            if i not in hash:
                return i
   
    else:
        return int(1)

firstMissingPositive([1,2,0])

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
class Solution:
    def mergeTwoLists(self, list1, list2):
        self.list1 = list1
        self.list2 = list2

        dummy = ListNode()
        tail = dummy 

        while list1 and list2:  
            # esto oti lista 1 kai lista 2 den einai 0 
            if list1.val < list2.val:
                #  thelo na valo proto noumero sthn oura, sth lista, to mikrotero noumero
                tail.next = list1
                list1 = list1.next 
            else:
                tail.next = list2
                list2 = list2.next
            
        if list1: 
            # ean mono h lista 1 den einai kenh []
            tail.next = list1
            list1 = list1.next
        
        elif list2: 
            # ean mono h lista 2 den einai kenh []
            tail.next = list2 
            list2 = list2.next 
        
        return dummy.next


lst1 = ListNode(1)
lst1.next = ListNode(2)
lst1.next.next = ListNode(3)    

lst2 = ListNode(1)
lst2.next = ListNode(3)
lst2.next.next = ListNode(4) 

sol = Solution()
print(sol.mergeTwoLists(lst1, lst2))

#uclidian Algorithm for GCD 

def euclidian_gcd():
    a = 106
    b =  6
    r=a%b
    while r:
        a=b
        b=r
        r=a%b
        return b

euclidian_gcd()

# Test Primality 

from math import sqrt

def prime(a, N):
    if a < N: 
        return False
    for x in range(N, int(sqrt(a)) + 1):
        print(x)
        if a % x == 0:
            return False
    return True

prime(3, 2)


def display_hash(hashTable):
	
	for i in range(len(hashTable)):
		print(i, end = " ")
		
		for j in hashTable[i]:
			print("-->", end = " ")
			print(j, end = " ")
			
		print()

HashTable = [[] for _ in range(10)]

def Hashing(keyvalue):
	return keyvalue % len(HashTable)

def insert(Hashtable, keyvalue, value):
	
	hash_key = Hashing(keyvalue)
	Hashtable[hash_key].append(value)

insert(HashTable, 10, 'Allahabad')
insert(HashTable, 25, 'Mumbai')
insert(HashTable, 20, 'Mathura')
insert(HashTable, 9, 'Delhi')
insert(HashTable, 21, 'Punjab')
insert(HashTable, 21, 'Noida')

display_hash (HashTable)

def make_hash_table():
	dicts = {}
	keyvalue = 10

	hashtable = [10,13,20,9,2,21,6,7,43,11]

	for index, y in enumerate(hashtable):
		results = keyvalue % y
		if y in dicts:
			dicts[results].append(y[index])
		else:
			dicts[results] = [y]

	return dicts 
make_hash_table()

#QuickSort O(nlogn) worst can be O(n2) --> time complexity and O(logn) --> space complexity

def partition(array):

    less = []
    equal = []
    greater = []

    if len(array) < 1:
        return []

    elif len(array) > 1:
        pivot = array[0]
        for x in array:
            if x < pivot:
                less.append(x)
            elif x == pivot:
                equal.append(x)
            elif x > pivot:
                greater.append(x)
      
        return partition(less) + equal + partition(greater)  
    else:
        return array

partition([8, 7, 2, 1, 0, 9, 6])

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def isSameTree(self, p, q) -> bool:
        self.p = p 
        self.q = q 

        if p is None and q is None:
            return True

        if p is not None and q is not None:
            return (p.val == q.val) and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        return False 

tree = TreeNode(1)
sec_tree = TreeNode(1)
tree.left = TreeNode(2)
tree.right = TreeNode(3)
sec_tree.left = TreeNode(2)
sec_tree.right = TreeNode(4)

if __name__ == "__main__":
    sol = Solution()
    if sol.isSameTree(tree, sec_tree):
        print("Both trees are identical")
    else:
        print("Trees are not identical")
	
def maxProfit(prices) -> int:
    
    smallest_num = min(prices)
    biggest_num = max(prices)
    day_1 = prices[0]

    if len(prices) < 2:
        return 0
    
    elif smallest_num == prices[-1] and biggest_num == prices[0]:
        return 0 
    
    elif smallest_num == prices[0]:
        return biggest_num - smallest_num

    elif smallest_num == prices[-1] and biggest_num != prices[0]:
        del prices[-1]
        return max(prices) - min(prices)
    
    else:
        for index, i in enumerate(prices):
            if smallest_num == i:
                return index 
            elif biggest_num == i and biggest_num == prices[0]:
                del prices[0]
            else: 
                return max(prices) - min(prices)

        return max(prices) - min(prices)
        
maxProfit([2,4,1])

def maxProfit(prices) -> int:
    left_ptr, profit = 0, 0

    for right_ptr in range(1, len(prices)):
      if prices[left_ptr] < prices[right_ptr]:
        profit = max(profit, prices[right_ptr] - prices[left_ptr])
      else:
        left_ptr = right_ptr

    return profit

maxProfit([4,3,1])

import re 
def mostCommonWord(paragraph, banned):

  paragraph = paragraph.lower()
  paragraph = re.sub(r'[^\w\s]',' ', paragraph)
  paragraph = paragraph.split()

  result = {}    
  for word in paragraph:                                                                                                                                                                                               
      result[word] = result.get(word, 0) + 1  
  
  for i in banned:
    if i in result:
      result.pop(i)
  
  if banned is not []:
  
    max_values = max(result.values())
    
    for key, value in result.items():
      if value == max_values:
        return key
  
  else:
    return ""
  
mostCommonWord('Bob hit a ball, the hit BALL flew far after it was hit.', ['hit'])

# DFS and Dynamic Programming with Probability Theory

class Solution:
    def knightProbability(self, n: int, k: int, row: int, column: int) -> float:
        from collections import deque

        matrix = [[0] * n for _ in range(n)]
        matrix[row][column] = 1

        if k == 1 and n == 1:
            return 0.0
        
        while k is not 0 or n is not 0:
            for _ in range(k):
                dp2 = [[0] * n for _ in range(n)]
                for r, row in enumerate(matrix):
                    for c, val in enumerate(row):
                        for dr, dc in ((2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)):
                            if 0 <= r + dr < n and 0 <= c + dc < n:
                                dp2[r+dr][c+dc] += val / 8.0
                                matrix = dp2

            return sum(map(sum, matrix))
        return 0.0

sol = Solution()
sol.knightProbability(7, 2, 0, 0)

import numpy as np 
def moveZeroes(nums) -> None:
  
  nums = sorted(nums)
  length_list = len(nums)
  
  final_dict = []
  
  if length_list == 0:
    return []

  elif length_list > 0:
    for i in range(length_list):
       if nums[i] == 0: 
        final_dict.append(nums[i])

  for i in range(len(final_dict)):
    if final_dict[i] <= nums[i]:
      nums.append(final_dict[i])
    else:
      nums[i] = final_dict[i]

  return np.trim_zeros(nums, 'f')

moveZeroes([0,1,0,3,12])

def intToRoman(num):

  val = [
            1000, 900, 500, 400,
            100, 90, 50, 40,
            10, 9, 5, 4,
            1
            ]
  syb = [
            "M", "CM", "D", "CD",
            "C", "XC", "L", "XL",
            "X", "IX", "V", "IV",
            "I"
            ]

  roman_number = ''
  i = 0
   
  while num > 0:
    for y in range(num // val[i]):
      roman_number += syb[i]
      num -= val[i]
    i += 1
  return roman_number
     
intToRoman(12)

from collections import Counter

def findCenter(edges) -> int:

  length_list = len(edges)
  
  flatten_list = [item for sublist in edges for item in sublist]
  
  if length_list is None:
    return []
  
  while length_list > 0:
    new_dict = {}
    for i in flatten_list:
      counting = flatten_list.count(i)
      new_dict[i] = counting
    
    for key, value in new_dict.items():
      if value == length_list:
        return key
        
findCenter([[1,2],[2,3],[4,2]])

def wordBreak(s: str, wordDict) -> bool:
    
    word_occurance = [True]
    for i in range(1, len(s)+1):
        result = False
        for j in range(i):
            if s[j:i] in wordDict and word_occurance[j]:
                result = True
        word_occurance.append(result)
    return word_occurance[-1]

wordBreak('catsandog', ["cats","dog","sand","and","cat"])

from collections import Counter
def longestPalindrome(s: str) -> int:

    if len(s) == 1:
        return 1
    
    elif len(s) == 0:
        return 0

    counter = Counter(s)
    final_result = []
    for key, value in counter.items():
        if value % 2 == 0:
            final_result.append(key * value)
        elif value % 2 != 0:
            final_result.append(key)

    for i in final_result:
        for y in final_result:
            for x in final_result:
                if len(i) > len(y) > 1 and len(x) == 1:
                    return int(len(y)/2 + len(i)/2 + len(x) + len(i)/2 + len(y)/2)
          

longestPalindrome("abccccdd")

def longestPalindrome(s: str) -> int:

    combinations = {}

    longest = 0
    for i in s:
        if i in combinations:
            combinations.pop(i)
            longest += 2
        else:
            combinations[i] = 1
 
    if combinations:
        longest+=1
    return longest
        

longestPalindrome("abccccdd")

import itertools 

def arrayPairSum(nums) -> int:

    sorted_list = sorted(nums)

    return sum(sorted_list[::2])    

arrayPairSum([6,2,6,5,1,2])

from collections import Counter
def firstUniqChar(s: str) -> int:
    
    if s == "":
        return 0 
    
    counter = Counter(s)
    string_to_list = list(s)
    
    my_list = []
    equal_list = []
    for key, val in counter.items():
        if val % 2 == 0:
            equal_list.append(key)
    
        elif val == 1:
            my_list.append(key)
    
        for index, val in enumerate(string_to_list):
            for idx, value in enumerate(my_list):
                    if value[0] in val:
                        return index
    return -1

firstUniqChar('llaa')

def kWeakestRows(mat, k: int):
    
    my_list = []
    my_dict = {}
    for key, value in enumerate(mat):
        sums = sum(value)
        my_dict[key] = sums
    my_dict = dict(sorted(my_dict.items(), key=lambda item: item[1]))
    final_list = list(my_dict.keys())
    return final_list[:k]
 
kWeakestRows([[1,1,0,0,0],[1,1,1,1,0],[1,0,0,0,0],[1,1,0,0,0],[1,1,1,1,1]], 3)

from collections import Counter
def isValidSudoku(board) -> bool:
    
    
    length = len(board)
    count = 0
    num = 0
    numbers = 0
    index = []
    status = True
    final_list = []
    
    lista_rows = []
    for indices in range(length):
        newlista  = [i for i in board[indices] if '.' not in i]
        lista_rows.append(newlista)
    
    def remove_empty_lists(lista):
        return (
            [remove_empty_lists(i) for i in lista if i!=[]]
            if isinstance(lista, list)
            else lista
        )

    final_lista_rows = remove_empty_lists(lista_rows)
    
    sorted_sets = []
    for sets in final_lista_rows:
        res = sorted(sets)
        sorted_sets.append(res)
   
    d = Counter([tuple(x) for x in sorted_sets])
    identicals = [list(k) for k, v in d.items() if v >= 2]
    flatten_identicals = [item for i in identicals for item in i ]
 
    flatten_identicals = [item for item, count in Counter(flatten_identicals).items() if count > 1]
    
    lists_2_rows = []
    for values in final_lista_rows:
        listtwo = dict((i, values.count(i)) for i in values)
        lists_2_rows.append(listtwo)
    

    false_true = []
    for vals in lists_2_rows:
        for keys, val in vals.items():
            if val > 1 or  len(flatten_identicals) > 1:
                false_true.append(False)
            elif flatten_identicals == [] or len(flatten_identicals) == 1:
                false_true.append(True)

    for i in board:
        count += length
        num += 1
        new = num - 1
        index.append(new)
    
    for y in index:
        for x in board:
            final_list.append(x[y])
    
    new_list = [final_list[i:i + length] for i in range(0, len(final_list), length)]
  
    list_to_dict = []
    for x in new_list:
        newlist  = [i for i in x if '.' not in i]
        list_to_dict.append(newlist)

    def remove_empty(lst):
        return (
            [remove_empty(i) for i in lst if i!=[]]
            if isinstance(lst, list)
            else lst
        )

    final_lista = remove_empty(list_to_dict)
    
    list_2 = []
    for values in final_lista:
        list2 = dict((i, values.count(i)) for i in values)
        list_2.append(list2)
    
    for vals in list_2:
        for keys, val in vals.items():
            for instances in false_true:
                if val > 1 or instances is False:
                    return False
    return True

isValidSudoku([[".",".",".",".","5",".",".","1","."],[".","4",".","3",".",".",".",".","."],[".",".",".",".",".","3",".",".","1"],["8",".",".",".",".",".",".","2","."],[".",".","2",".","7",".",".",".","."],[".","1","5",".",".",".",".",".","."],[".",".",".",".",".","2",".",".","."],[".","2",".","9",".",".",".",".","."],[".",".","4",".",".",".",".",".","."]])


select
MAX(score_points) as highest_points
from 
    (select distinct ha.name as name, 
    ha.hacker_id as ha_id, 
    sub.score,
        SUM(CASE WHEN cha.difficulty_level = 1 and sub.score >=  20 THEN 1 
                 WHEN cha.difficulty_level = 2 and sub.score >=  30 THEN 1 
                 WHEN cha.difficulty_level = 3 and sub.score >=  40 THEN 1 
                 WHEN cha.difficulty_level = 4 and sub.score >=  60 THEN 1 
                 WHEN cha.difficulty_level = 5 and sub.score >=  80 THEN 1 
                 WHEN cha.difficulty_level = 6 and sub.score >=  100 THEN 1 
                 WHEN cha.difficulty_level = 7 and sub.score >=  120 THEN 1 ELSE 0 END) as score_points
    from Hackers as ha
    inner join Challenges as cha 
    on cha.hacker_id = ha.hacker_id
    inner join Submissions as sub
    on ha.hacker_id = sub.hacker_id
    group by ha.name, ha.hacker_id, sub.score) 
max_grade
;

with final_table as (select genre.genre, 
       max(Domestic + Worldwide - Budget) as net_profit
from imdb
left join genre
    on genre.Movie_id = imdb.Movie_id
left join earning 
    on imdb.Movie_id = earning.Movie_id
where imdb.Title like '%(2012)%' and genre.genre is not null
group by genre.genre)
select genre, max(net_profit) as net_profit
from final_table
where net_profit is not null
group by genre
order by genre;

select
COUNT(*) as ConsecutiveNums from (select Id as Id, 
    row_number() over (partition by Id) - row_number() over (partition by Num order by ID) as 
            ConsecutiveNums
from Logs) Logs
where Id = (select max(Id) from Logs);

select Department, Employee, Salary from (
               select 
               dep.name as Department, emp.name as Employee, emp.Salary,
               rank() over (partition by dep.name order by Salary desc) rn
from Employee as emp
left join Department as dep 
on emp.DepartmentId = dep.Id) T
where rn = 1
order by Employee desc;

with final_table as (select * from (select
case when ac.income <= 20000 then 'Low Salary'
     when ac.income between 20000 and 50000 then 'Average Salary'
     when ac.income >= 50000 then 'High Salary'
     else '' end as category,
     count( ac.account_id) as accounts_count
from Accounts as ac
group by ac.income) t)
select category, count(*) as accounts_count
from final_table
group by category
order by accounts_count
;

SELECT
'Low Salary' AS category,
COUNT(*) AS accounts_count
FROM Accounts
WHERE income<20000
union all 
select 
'Average Salary' as category,
count(*) as accounts_count
from Accounts
where income between 20000 and 50000
union all 
select
'Highest Salary' as category,
count(*) as accounts_count
from Accounts
where income>50000;
