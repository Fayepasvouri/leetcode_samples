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
