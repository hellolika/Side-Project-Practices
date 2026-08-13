CREATE PROCEDURE [dbo].[HRMS_LeaveRequestApproval.2.0.1]
	@approverId INT,
	@requestId INT,
	@isApproved INT,
	@responseReason nvarchar(200)
AS
BEGIN
	--THIS SP WILL NOT TRIGGERD WHEN LEAVE RECORD IS PENDING
	SET NOCOUNT ON;
	IF EXISTS (SELECT TOP 1 1 FROM [dbo].[TakeLeaveRecord] WITH(NOLOCK) WHERE [RequestId] = @requestId)
	BEGIN
		--IF((SELECT [Status] FROM [dbo].[TakeLeaveRecord] WITH(NOLOCK) WHERE [RequestId] = @requestId) = 1)
		--BEGIN
		--	SELECT 21 AS ErrorCode;
		--	RETURN;
		--END
		--IF((SELECT [Status] FROM [dbo].[TakeLeaveRecord] WITH(NOLOCK) WHERE [RequestId] = @requestId) = 2)
		--BEGIN
		--	SELECT 22 AS ErrorCode;
		--	RETURN;
		--END
		IF((SELECT [IsCancel] FROM [dbo].[TakeLeaveRecord] WITH(NOLOCK) WHERE [RequestId] = @requestId) = 1)
		BEGIN
			SELECT 27 AS ErrorCode;
			RETURN;
		END

		DECLARE @oldApprovalStatus AS INT
		DECLARE @memberId AS INT
		DECLARE @leaveTypeId AS INT

		SELECT @oldApprovalStatus = [Status], @memberId = [MemberId], @leaveTypeId = [LeaveType]
		FROM [dbo].[TakeLeaveRecord] WHERE [RequestId] = @requestId


		--CHECKING IF ADMIN REQUEST THE SAME STATUS
			--CHECKING IF ALREADY APRROVED
		IF(@isApproved = @oldApprovalStatus AND @isApproved = 1)
		BEGIN 
			SELECT 21 AS ErrorCode;
			RETURN;
		END
			--CHECKING IF ALREADY REJECTED
		ELSE IF(@isApproved = @oldApprovalStatus AND @isApproved = 2)
		BEGIN 
			SELECT 22 AS ErrorCode;
			RETURN;
		END

		DECLARE @remainingLeave AS DECIMAL(16,9)
		DECLARE @leaveRequested AS DECIMAL(16,9)

		IF(@isApproved = 1)
		BEGIN
			--CHECKING IF ADMIN EDIT FROM REJECTED TO APROVED
			IF(@oldApprovalStatus = 2)
				--DEDUCTING BACK THE LEAVE REMAINING
				BEGIN
					SELECT @remainingLeave	=	[RemainingLeaves] FROM [dbo].[LeaveAmount] 
											WHERE [MemberId] = @memberId AND [LeaveType] = @leaveTypeId;

					SELECT @leaveRequested	=	[NumberOfDay] FROM [dbo].[TakeLeaveRecord] WITH(NOLOCK) 
											WHERE [RequestId] = @requestId;
					UPDATE [dbo].[LeaveAmount] 
					SET [RemainingLeaves] = @remainingLeave - @leaveRequested  
					WHERE [MemberId] = @memberId
					AND [LeaveType] = @leaveTypeId;
				END

				BEGIN 
					UPDATE [dbo].[TakeLeaveRecord] 
					SET [Status] = 1, [ResponseReason] = @responseReason,
					[UpdateBy] = @approverId, [UpdateOn] = GETDATE()
					WHERE [RequestId] = @requestId;
				END
			BEGIN 
				UPDATE [dbo].[TakeLeaveRecord] 
				SET [Status] = 1, [ResponseReason] = @responseReason,
				[UpdateBy] = @approverId, [UpdateOn] = GETDATE()
				WHERE [RequestId] = @requestId;
			END
		END
		ElSE
		BEGIN

			--THIS CODE BELOW TRIES TO ADD BACK THE LEAVE TO REMAINING LEAVE BUT WE ALREADY DID THAT IN UPDATE TAKE LEAVE
			SELECT @remainingLeave	=	[RemainingLeaves] FROM [dbo].[LeaveAmount] l WITH(NOLOCK) 
										INNER JOIN [dbo].[TakeLeaveRecord] t WITH(NOLOCK)
										ON l.[MemberId] = t.[MemberId] AND l.[LeaveType] = t.[LeaveType] 
										WHERE t.[RequestId] = @requestId AND l.[Year] = YEAR(t.[StartDate]);

			SELECT @leaveRequested	=	[NumberOfDay] FROM [dbo].[TakeLeaveRecord] WITH(NOLOCK) 
										WHERE [RequestId] = @requestId;

			UPDATE [dbo].[LeaveAmount] 
			SET [RemainingLeaves] = @remainingLeave + @leaveRequested  
			WHERE [MemberId] = @memberId
			AND [LeaveType] = @leaveTypeId;
			
			UPDATE [dbo].[TakeLeaveRecord] 
			SET [Status] = 2, [ResponseReason] = @responseReason , [UpdateBy] = @approverId
			WHERE [RequestId] = @requestId;
			
		END

		SELECT 0 AS ErrorCode;
	END

	ELSE
	BEGIN
		SELECT 11 AS ErrorCode;
	END
END
