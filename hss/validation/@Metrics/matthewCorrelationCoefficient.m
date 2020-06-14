function matthewCorrelationCoefficient(obj)
  num = obj.TP*obj.TN-obj.FP*obj.FN;
  den = sqrt((obj.TP+obj.FP)*(obj.TP+obj.FN)*(obj.TN+obj.FP)*(obj.TN+obj.FN));
  obj.MCC = num/den;
end