def GenerateTrainVector(self, ID, maxID, lr_matrix, reg_rate_matrix):
    SFGSS = 8
    SFHC = 20
    Fl = 0.1
    Fu = 0.9
    tuo1 = 0.1
    tuo2 = 0.03
    tuo3 = 0.07

    Result = np.empty(shape=(2, 1))

    u1 = ID
    u2 = ID
    u3 = ID
    while (u1 == ID):
        a = np.random.rand() * (maxID - 1)
        u1 = round(a)
    while ((u2 == ID) or (u2 == u1)):
        b = np.random.rand() * (maxID - 1)
        u2 = round(b)
    while ((u3 == ID) or (u3 == u2) or (u3 == u1)):
        c = np.random.rand() * (maxID - 1)
        u3 = round(c)

    rand1 = np.random.rand()
    rand2 = np.random.rand()
    rand3 = np.random.rand()
    F = np.random.rand()
    K = np.random.rand()

    if (rand3 < tuo2):
        F = SFGSS
    elif (tuo2 <= rand3 and rand3 < tuo3):
        F = SFHC
    elif (rand2 < tuo1 and rand3 > tuo3):
        F = Fl + Fu * rand1

    temp1 = lr_matrix[u2][0] - lr_matrix[u3][0]
    temp2 = temp1 * F
    temp_mutation = lr_matrix[u1][0] + temp2
    temp1 = temp_mutation - lr_matrix[ID][0]
    temp2 = temp1 * K
    Result[0][0] = lr_matrix[ID][0] + temp2

    temp1 = reg_rate_matrix[u2][0] - reg_rate_matrix[u3][0]
    temp2 = temp1 * F
    temp_mutation = reg_rate_matrix[u1][0] + temp2
    temp1 = temp_mutation - reg_rate_matrix[ID][0]
    temp2 = temp1 * K
    Result[1][0] = reg_rate_matrix[ID][0] + temp2

    if (Result[0][0] <= self.lr_min):
        Result[0][0] = self.lr_min
    if (Result[0][0] >= self.lr_max):
        Result[0][0] = self.lr_max
    if (Result[1][0] <= self.reg_rate_min):
        Result[1][0] = self.reg_rate_min
    if (Result[1][0] >= self.reg_rate_max):
        Result[1][0] = self.reg_rate_max

    return Result