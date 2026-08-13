CREATE PROCEDURE [dbo].[HRMS_GetMemberTransfers.1.0.0]
  @memberId INT
AS
BEGIN
  SELECT tf.Id as TransferId, tt.TransferName, tf.MemberId, m.Username as MemberName, tf.IsGenerated, tf.TransferTypeId , tf.DateCount,tt.BeneficiaryTypeId, tf.Amount, tf.Remark, tf.PayDate, tf.PayStartDate, tf.PayEndDate, tf.CreatedOn
  FROM Transfer tf
  JOIN TransferType tt
  ON tf.TransferTypeId = tt.Id 
  JOIN Member m 
  ON tf.MemberId = m.Id
  WHERE tf.MemberId = @memberId AND tf.[Status] = 2
END
