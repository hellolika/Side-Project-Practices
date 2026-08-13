CREATE PROCEDURE [dbo].[HRMS_TakeLeave.1.1.0]
	@memberId INT,
	@numberOfDay AS DECIMAL(16,9),
	@startDate DATETIME,
	@endDate DATETIME,
	@leaveType INT,
	@image VARCHAR(100),
	@reason NVARCHAR(500)
AS
BEGIN
	SET NOCOUNT ON;
	IF(YEAR(@startDate) <> YEAR(@endDate))
	BEGIN
		SELECT 24 AS ErrorCode;
		RETURN;
	END
	
	-- CHECKING ARE THERE PENDING LEAVE OR NOT
	IF EXISTS(SELECT TOP 1 1 FROM [dbo].[TakeLeaveRecord] WHERE [MemberId] = @memberId AND 
	[Status] IN (0, 1) AND [IsCancel] = 0 AND
	((@startDate BETWEEN [StartDate] AND [EndDate]) OR (@endDate BETWEEN [StartDate] AND [EndDate])
	OR ([StartDate] BETWEEN @startDate AND @endDate) OR ([EndDate] BETWEEN @startDate AND @endDate)))
	BEGIN
		SELECT 30 AS ErrorCode; 
		RETURN;
	END
	
	DECLARE @isLimited AS BIT
	SELECT @isLimited = [IsLimited] FROM [dbo].[LeaveType] WITH(NOLOCK) WHERE [TypeId] = @leaveType AND [IsEnable] = 1;

	IF (@isLimited IS NOT NULL)
	BEGIN
		IF(@isLimited = 1)
		BEGIN
			DECLARE @leaveAmount AS DECIMAL(16,9);

			SELECT @leaveAmount = [RemainingLeaves] FROM [dbo].[LeaveAmount] WITH(NOLOCK) 
			WHERE [MemberId] = @memberId AND [LeaveType] = @leaveType
			AND [Year] = YEAR(@startDate)
			
			-- CHECKING IF MEMBER STILL HAVE REMAINING LEAVES
			IF(@leaveAmount IS NULL OR @leaveAmount - @numberOfDay < 0)
			BEGIN
				SELECT 23 AS ErrorCode;
				RETURN;
			END
			
			UPDATE [dbo].[LeaveAmount] 
			SET [RemainingLeaves] = @leaveAmount - @numberOfDay  
			WHERE [MemberId] = @memberId AND [LeaveType] = @leaveType
		END

		INSERT INTO [dbo].[TakeLeaveRecord] ([MemberId], [NumberOfDay], [StartDate], [EndDate], [LeaveType],[Image], [Reason])
		OUTPUT 0 AS ErrorCode, INSERTED.RequestId
		VALUES (@memberId, @numberOfDay, @startDate, @endDate, @leaveType, @image, @reason);
	END
	ELSE
	BEGIN
		SELECT 13 AS ErrorCode;
	END
END
