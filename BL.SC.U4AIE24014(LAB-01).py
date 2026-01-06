import random

def cnt(text):
	vowels="aeiouAEIOU"
	v=c=0
	for ch in text:
		if ch.isalpha():
			if ch in vowels:
				v+=1
			else:
				c+=1
	return v,c

def matrices(matA, matB):
	if len(matA[0]) != len(matB):
		return "error...not multilpicable"

	res=[[0 for x in range(len(matB[0]))] for x in range(len(matA))]

	for i in range(len(matA)):
		for j in range(len(matB[0])):
			for k in range(len(matB)):
				res[i][j]+= matA[i][k]*matB[k][j]
	return res

def list_common(l1,l2):
	common=0
	for i in set(l1):
		if i in set(l2):
			common+=1
	return common

def transpose(mat):
	r=len(mat)
	c=len(mat[0])
	transpose=[[0 for _ in range(r)] for _ in range(c)]

	for i in range(r):
		for j in range(c):
			transpose[j][i]=mat[i][j]
	return transpose

def cal():
	lst=[random.randint(100,150) for _ in range(100)]
	lst.sort()

	mean = sum(lst)/len(lst)
	median = (lst[49] + lst[50]) / 2

	freq = {}
	for num in lst:
        	freq[num] = freq.get(num, 0) + 1

	mode = max(freq, key=freq.get)
	return mean, median, mode

print("Vowels and consonents: ", cnt("Machine learning"))
A=[[1,2],[3,4]]
B=[[5,6],[7,8]]
print("Product of matrices: ", matrices(A,B))
print("Common elements: ", list_common([1,2,3,4,4,5,6,4,3,5,2,9,6,7,8],[3,5,6,5,4,3,2,1,9,0,8,7,6,7,8]))
print("Transpose: ", transpose(A))
mean, median, mode=cal()
print("Mean: ", mean, "Median: ", median, "Mode: ", mode)