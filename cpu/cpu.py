from cards import Cards
from learn import Model


class Doudizhu(Cards):
    def __init__(self, take):
        super(Doudizhu, self).__init__(take)
        self.brain = Model()

    def judge(self, legal_now):
        '''选取最合适的牌型'''
        now = legal_now.typ
        max_point = -99999.0
        otp = None
        for i in self.relation[(now[0], now[1], now[3])]:
            if i.card_array[0] > legal_now.card_array[0]:
                point = self.brain.model.predict(self.get_after(i)))
                if point > max_point:
                    max_point = point
                    otp = i
        return otp, max_point
