CREATE PROCEDURE [dbo].[HRMS_UpdateMemberMonthlyTransferStatus.1.0.0]
  @transferId INT,
  @status INT
AS
BEGIN
  UPDATE [dbo].[Transfer] SET [Status] = @status WHERE Id = @transferId;
  SELECT 0 AS ErrorCode;
END

-- Stauts: 
-- 0 => Pending
-- 1 => Confirmed
-- 2 = > Sent

-- EXEC [dbo].[HRMS_UpdateMemberMonthlyTransferStatus.1.0.0]
-- @transferId = 2013,
-- @status = 2

