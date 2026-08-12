CREATE PROCEDURE [dbo].[HRMS_AddOrEditLeaveRecord.1.0.0]
  @leaveRecordId INT = 0,
  @memberId INT,
	@numberOfDay DECIMAL(16,9),
	@startDate DATETIME,
	@endDate DATETIME,
	@leaveType INT
AS
IF EXISTS (SELECT TOP 1 1 FROM [dbo].[TakeLeaveRecord] WHERE RequestId = @leaveRecordId)
  BEGIN
    UPDATE [dbo].[TakeLeaveRecord] SET [MemberId] = @memberId, [NumberOfDay] = @numberOfDay, [StartDate] = @startDate,
    [EndDate] = @endDate, [LeaveType] = @leaveType
    WHERE RequestId = @leaveRecordId
    SELECT 0 AS ErrorCode;
  END
ELSE
  BEGIN
    INSERT INTO [dbo].[TakeLeaveRecord] ([MemberId], [NumberOfDay], [StartDate], [EndDate], [LeaveType]) 
    VALUES (@memberId, @numberOfDay, @startDate, @endDate, @leaveType);

    SELECT 0 AS ErrorCode;
  END


