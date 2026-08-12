CREATE PROCEDURE [dbo].[HRMS_GetMonthlyTransfers.1.0.0]
  @startDate DATETIME,
  @endDate DATETIME
AS
BEGIN
    SELECT tf.Id as TransferId, tt.TransferName, tf.MemberId, m.Username as MemberName, m.Email, m.BankName, m.BankAccount, m.PhoneNumber,m.[Position], t.TeamName as TeamName, tf.TransferTypeId ,tt.[BeneficiaryTypeId], tf.status, tf.DateCount, tf.Amount, tf.Remark, tf.PayDate, tf.PayStartDate, tf.PayEndDate, tf.IsGenerated, tf.CreatedOn
  FROM Transfer tf
  JOIN TransferType tt 
  ON tf.TransferTypeId = tt.Id
  JOIN Member m 
  ON tf.MemberId = m.Id
  LEFT JOIN Member md 
  ON tf.ModifiedBy = md.Id
  JOIN Team t 
  ON t.TeamId = m.TeamId
  WHERE PayDate BETWEEN @startDate AND @endDate
  
END