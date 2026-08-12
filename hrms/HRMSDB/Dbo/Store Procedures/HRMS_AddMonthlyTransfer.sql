CREATE PROCEDURE [dbo].[HRMS_AddMonthlyTransfer.1.0.0]
  @transferId INT = 0,
  @memberId INT,
  @propertyTypeId INT,
  @amount DECIMAL(10,2),
  @dayCount INT,
  @status INT,
  @remark NVARCHAR(1000),
  @payRollDate DATETIME,
  @startDate DATETIME,
  @endDate DATETIME,
  @createdBy INT,
  @modifiedBy INT
AS
IF EXISTS (SELECT TOP 1 1 FROM [dbo].[Transfer] WHERE Id = @transferId)
  BEGIN
    UPDATE [dbo].[Transfer] SET TransferTypeId = @propertyTypeId, Amount = @amount, Status = @status, Remark = @remark,PayStartDate = @startDate, PayEndDate = @endDate, ModifiedBy = @modifiedBy, ModifiedOn = GETDATE() WHERE Id = @transferId
    SELECT 0 AS ErrorCode;
  END
ELSE
  IF(@propertyTypeId = 1)
  BEGIN
    IF EXISTS (SELECT TOP 1 1 FROM [dbo].[Transfer] WHERE MemberId = @memberId AND TransferTypeId = 1 AND MONTH(@startDate) = MONTH(PayStartDate) AND YEAR(@startDate) = YEAR(PayStartDate) AND MONTH(@startDate) = MONTH(PayEndDate) AND YEAR(@startDate) = YEAR(PayEndDate))
    BEGIN
        SELECT 42 AS ErrorCode
    END
     ELSE 
    BEGIN
      INSERT INTO [dbo].[Transfer] ([MemberId], [TransferTypeId],[DateCount], [Amount],[Status], [Remark], [PayDate],[PayStartDate],[PayEndDate],[CreatedBy], [ModifiedBy]) 
      VALUES (@memberId, @propertyTypeId,@dayCount, @amount,@status, @remark, @payRollDate,@startDate,@endDate, @createdBy, @modifiedBy);

      SELECT 0 AS ErrorCode;
    END
  END
  ELSE 
  BEGIN
    INSERT INTO [dbo].[Transfer] ([MemberId], [TransferTypeId],[DateCount], [Amount],[Status], [Remark], [PayDate],[PayStartDate],[PayEndDate],[CreatedBy], [ModifiedBy]) 
    VALUES (@memberId, @propertyTypeId,@dayCount, @amount,@status, @remark, @payRollDate,@startDate,@endDate, @createdBy, @modifiedBy);

    SELECT 0 AS ErrorCode;
  END