import random


nations=['汉','蒙古','汉','汉','蒙古','蒙古','汉','汉','回','回','回','藏','藏','维吾尔','维吾尔','苗','苗''蒙古','汉','汉','汉','彝','蒙古','壮','布依','朝鲜','满','侗','汉','汉','瑶','白','土家','哈尼','哈萨克','傣','黎','僳僳','佤','畲','高山','拉祜','汉','汉','水','东乡','汉','纳西','景颇','柯尔克孜','土','达斡尔','仫佬','羌','布朗','撒拉','毛南','仡佬','锡伯','阿昌','普米','塔吉克','怒','乌孜别克','俄罗斯','鄂温克','德昂','保安','裕固','京','塔塔尔','独龙','鄂伦春','赫哲','门巴','珞巴','基诺']

def sex_nation():
    temp=random.choice(range(5))
    if temp==0:
        return '性 别'+" "*random.randint(1,2)+'男'+" "*random.randint(2,3)+'民 族'+" "*random.randint(1,2)+random.choice(nations)
    elif temp==1:
        return '性 别'+" "*random.randint(1,2)+'男'
    elif temp==2:
        return '民 族'+" "*random.randint(1,2)+random.choice(nations)
    elif temp==3:
        return  '性 别'+" "*random.randint(1,2)+'女'+" "*random.randint(2,3)+'民 族'+" "*random.randint(1,2)+random.choice(nations)
    else:
        return '性 别'+" "*random.randint(1,2)+'女'
def brithday():
    #所有月份都按31天算，不算闰年
    temp=random.choice(range(10))
    if temp>3:
        year=random.randint(1920,1970)
    else:
        year=random.randint(1970,2020)
    month=random.randint(1,12)
    day=random.randint(1,31)
    return random.choice(["","出 生"])+str(year)+" 年 "+str(month)+" 月 "+str(day)+" 日"
    
def data_limit():
    temp=random.choice(range(10))
    if temp>3:
        year=random.randint(1920,1970)
    else:
        year=random.randint(1970,2020)
    month=random.randint(1,12)
    day=random.randint(1,31)
    if random.choice(range(2))>0:
        return random.choice(["","有效期限"+" "*random.randint(1,2)])+str(year)+"."+str(month)+"."+str(day)+"-"+str(year+random.choice([5,10,20]))+"."+str(month)+"."+str(day)
    else:
        return str(year)+"."+str(month)+"."+str(day)+"-"+str(year+random.choice([5,10,20]))+"."+str(month)+"."+str(day)
