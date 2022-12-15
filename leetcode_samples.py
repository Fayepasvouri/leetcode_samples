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
